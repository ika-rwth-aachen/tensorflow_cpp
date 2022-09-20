/*
==============================================================================
MIT License
Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================
*/

/**
 * @file
 * @brief Model class
 */

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow_cpp/graph_utils.h>
#include <tensorflow_cpp/saved_model_utils.h>
#include <tensorflow_cpp/utils.h>

/**
 * @brief Namespace for tensorflow_cpp library
 */
namespace tensorflow_cpp {


/**
 * @brief Wrapper class for running TensorFlow SavedModels or FrozenGraphs.
 */
class Model {

 public:
  /**
   * @brief Creates an uninitialized model.
   */
  Model() {}

  /**
   * @brief Creates a model by loading it from disk.
   *
   * @param[in]  model_path                       SavedModel or FrozenGraph path
   * @param[in]  warmup                           run dummy inference to warmup
   * @param[in]  allow_growth                     dynamically grow GPU usage
   * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
   * @param[in]  visible_device_list              list of GPUs to use, e.g.
   * "0,1"
   */
  Model(const std::string& model_path, const bool warmup = false,
        const bool allow_growth = true,
        const double per_process_gpu_memory_fraction = 0,
        const std::string& visible_device_list = "") {

    loadModel(model_path, warmup, allow_growth, per_process_gpu_memory_fraction,
              visible_device_list);
  }

  /**
   * @brief Loads a SavedModel or FrozenGraph model from disk.
   *
   * After the model has loaded, it's also run once with dummy inputs in order
   * to speed-up the first actual inference call.
   *
   * @param[in]  model_path                       SavedModel or FrozenGraph path
   * @param[in]  warmup                           run dummy inference to warmup
   * @param[in]  allow_growth                     dynamically grow GPU usage
   * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
   * @param[in]  visible_device_list              list of GPUs to use, e.g.
   * "0,1"
   */
  void loadModel(const std::string& model_path, const bool warmup = false,
                 const bool allow_growth = true,
                 const double per_process_gpu_memory_fraction = 0,
                 const std::string& visible_device_list = "") {

    is_frozen_graph_ = (model_path.substr(model_path.size() - 3) == ".pb");
    is_saved_model_ = !is_frozen_graph_;

    // load model
    if (is_frozen_graph_) {
      graph_def_ = loadFrozenGraph(model_path);
      session_ = createSession(allow_growth, per_process_gpu_memory_fraction,
                               visible_device_list);
      loadGraphIntoSession(session_, graph_def_);
    } else {
      saved_model_ =
        loadSavedModel(model_path, allow_growth,
                       per_process_gpu_memory_fraction, visible_device_list);
      session_ = saved_model_.GetSession();
    }

    // automatically find inputs and outputs
    if (is_frozen_graph_) {
      input_names_ = getGraphInputNames(graph_def_);
      output_names_ = getGraphOutputNames(graph_def_);
    } else {
      input_names_ = getSavedModelInputNames(saved_model_, true);
      output_names_ = getSavedModelOutputNames(saved_model_, true);
      const auto input_nodes_ = getSavedModelInputNames(saved_model_, false);
      const auto output_nodes_ = getSavedModelOutputNames(saved_model_, false);
      for (int k = 0; k < input_names_.size(); k++) {
        saved_model_node2layer_[input_nodes_[k]] = input_names_[k];
        saved_model_layer2node_[input_names_[k]] = input_nodes_[k];
      }
      for (int k = 0; k < output_names_.size(); k++) {
        saved_model_node2layer_[output_nodes_[k]] = output_names_[k];
        saved_model_layer2node_[output_names_[k]] = output_nodes_[k];
      }
    }
    n_inputs_ = input_names_.size();
    n_outputs_ = output_names_.size();

    // run dummy inference to warm-up
    if (warmup) dummyCall();
  }

  /**
   * @brief Checks whether the model is loaded already.
   *
   * @return  true   if model is loaded
   * @return  false  if model is not loaded
   */
  bool isLoaded() const {

    bool is_loaded = bool(session_);

    return is_loaded;
  }

  /**
   * @brief Runs the model.
   *
   * The input/output names are expected to be set to the model layer names
   * given during model construction. Information about the model can be
   * printed using `getInfoString`. For FrozenGraphs, layer names are unknown
   * and node names are expected.
   *
   * @param[in]  inputs                                       inputs by name
   * @param[in]  output_names                                 output names
   *
   * @return  std::unordered_map<std::string, tf::Tensor>     outputs by name
   */
  std::unordered_map<std::string, tf::Tensor> operator()(
    const std::vector<std::pair<std::string, tf::Tensor>>& inputs,
    const std::vector<std::string>& output_names) {

    // properly set input/output names for session->Run()
    std::vector<std::pair<std::string, tf::Tensor>> input_nodes;
    std::vector<std::string> output_node_names;
    if (is_saved_model_) {
      for (const auto& input : inputs)
        input_nodes.push_back(
          {saved_model_layer2node_[input.first], input.second});
      for (const auto& name : output_names)
        output_node_names.push_back(saved_model_layer2node_[name]);
    } else if (is_frozen_graph_) {
      input_nodes = inputs;
      output_node_names = output_names;
    } else {
      return {};
    }

    // run model
    tf::Status status;
    std::vector<tf::Tensor> output_tensors;
    status = session_->Run(input_nodes, output_node_names, {}, &output_tensors);

    // build outputs
    std::unordered_map<std::string, tf::Tensor> outputs;
    if (status.ok()) {
      for (int k = 0; k < output_tensors.size(); k++)
        outputs[output_names[k]] = output_tensors[k];
    } else {
      throw std::runtime_error("Failed to run model: " + status.ToString());
    }

    return outputs;
  }

  /**
   * @brief Runs the model.
   *
   * This version of `operator()` works without having to specify input/output
   * names of the model, but is limited to single-input/single-output models.
   *
   * @param[in]  input_tensor  input tensor
   *
   * @return  tf::Tensor       output tensor
   */
  tf::Tensor operator()(const tf::Tensor& input_tensor) {

    if (n_inputs_ != 1 || n_outputs_ != 1) {
      throw std::runtime_error(
        "'tf::Tensor tensorflow_cpp::Model::operator()(const tf::Tensor&)' is "
        "only available for single-input/single-output models. Found " +
        std::to_string(n_inputs_) + " inputs and " +
        std::to_string(n_outputs_) + " outputs.");
    }

    // run model
    auto outputs =
      (*this)({{input_names_[0], input_tensor}}, {output_names_[0]});

    return outputs[output_names_[0]];
  }

  /**
   * @brief Runs the model.
   *
   * This version of `operator()` works without having to specify input/output
   * names of the model, but is limited to FrozenGraph models.
   *
   * @param[in]  input_tensors            input tensors
   *
   * @return  std::vector<tf::Tensor>     output tensors
   */
  std::vector<tf::Tensor> operator()(
    const std::vector<tf::Tensor>& input_tensors) {

    if (input_tensors.size() != n_inputs_) {
      throw std::runtime_error(
        "Model has " + std::to_string(n_inputs_) + " inputs, but " +
        std::to_string(input_tensors.size()) + " input tensors were given");
    }

    // assign inputs in default order
    std::vector<std::pair<std::string, tf::Tensor>> inputs;
    for (int k = 0; k < n_inputs_; k++)
      inputs.push_back({input_names_[k], input_tensors[k]});

    // run model
    auto outputs = (*this)(inputs, output_names_);

    // return output tensors in default order
    std::vector<tf::Tensor> output_tensors;
    for (const auto& name : output_names_)
      output_tensors.push_back(outputs[name]);

    return output_tensors;
  }

  /**
   * @brief Determines the shape of a model node.
   *
   * @param[in]  name              node name
   *
   * @return  std::vector<int>     node shape
   */
  std::vector<int> getNodeShape(const std::string& name) {

    if (is_saved_model_) {
      return getSavedModelNodeShape(saved_model_,
                                    saved_model_layer2node_[name]);
    } else if (is_frozen_graph_) {
      return getGraphNodeShape(graph_def_, name);
    } else {
      return {};
    }
  }

  /**
   * @brief Determines the shape of the model input.
   *
   * This function works without having to specify input/output names of the
   * model, but is limited to single-input/single-output models.
   *
   * @return  std::vector<int>  node shape
   */
  std::vector<int> getInputShape() {

    if (n_inputs_ != 1) {
      throw std::runtime_error(
        "std::vector<int> tensorflow_cpp::Model::getInputShape()' is only "
        "available for single-input models. Found " +
        std::to_string(n_inputs_) + " inputs.");
    }

    return getNodeShape(input_names_[0]);
  }

  /**
   * @brief Determines the shape of the model output.
   *
   * This function works without having to specify input/output names of the
   * model, but is limited to single-input/single-output models.
   *
   * @return  std::vector<int>  node shape
   */
  std::vector<int> getOutputShape() {

    if (n_outputs_ != 1) {
      throw std::runtime_error(
        "std::vector<int> tensorflow_cpp::Model::getOutputShape()' is only "
        "available for single-output models. Found " +
        std::to_string(n_outputs_) + " outputs.");
    }

    return getNodeShape(output_names_[0]);
  }

  /**
   * @brief Determines the shape of the model inputs.
   *
   * @return  std::vector<std::vector<int>>  node shapes
   */
  std::vector<std::vector<int>> getInputShapes() {

    std::vector<std::vector<int>> shapes;
    for (const auto& name : input_names_) shapes.push_back(getNodeShape(name));

    return shapes;
  }

  /**
   * @brief Determines the shape of the model outputs.
   *
   * @return  std::vector<std::vector<int>>  node shapes
   */
  std::vector<std::vector<int>> getOutputShapes() {

    std::vector<std::vector<int>> shapes;
    for (const auto& name : output_names_) shapes.push_back(getNodeShape(name));

    return shapes;
  }

  /**
   * @brief Determines the datatype of a model node.
   *
   * @param[in]  name          node name
   *
   * @return  tf::DataType     node datatype
   */
  tf::DataType getNodeType(const std::string& name) {

    if (is_saved_model_) {
      return getSavedModelNodeType(saved_model_, saved_model_layer2node_[name]);
    } else if (is_frozen_graph_) {
      return getGraphNodeType(graph_def_, name);
    } else {
      return tf::DataType();
    }
  }

  /**
   * @brief Determines the datatype of the model input.
   *
   * This function works without having to specify input/output names of the
   * model, but is limited to single-input/single-output models.
   *
   * @return  tf::DataType  node datatype
   */
  tf::DataType getInputType() {

    if (n_inputs_ != 1) {
      throw std::runtime_error(
        "'tf::DataType tensorflow_cpp::Model::getInputType()' is only "
        "available for single-input models. Found " +
        std::to_string(n_inputs_) + " inputs.");
    }

    return getNodeType(input_names_[0]);
  }

  /**
   * @brief Determines the datatype of the model output.
   *
   * This function works without having to specify input/output names of the
   * model, but is limited to single-input/single-output models.
   *
   * @return  tf::DataType  node datatype
   */
  tf::DataType getOutputType() {

    if (n_outputs_ != 1) {
      throw std::runtime_error(
        "'tf::DataType tensorflow_cpp::Model::getOutputType()' is only "
        "available for single-output models. Found " +
        std::to_string(n_outputs_) + " outputs.");
    }

    return getNodeType(output_names_[0]);
  }

  /**
   * @brief Determines the datatype of the model inputs.
   *
   * @return  std::vector<tf::DataType>  node datatypes
   */
  std::vector<tf::DataType> getInputTypes() {

    std::vector<tf::DataType> types;
    for (const auto& name : input_names_) types.push_back(getNodeType(name));

    return types;
  }

  /**
   * @brief Determines the datatype of the model outputs.
   *
   * @return  std::vector<tf::DataType>  node datatypes
   */
  std::vector<tf::DataType> getOutputTypes() {

    std::vector<tf::DataType> types;
    for (const auto& name : output_names_) types.push_back(getNodeType(name));

    return types;
  }

  /**
   * @brief Returns information about the model.
   *
   * Returns a formatted message containing information about the shape and type
   * of all inputs/outputs of the model.
   *
   * @return  std::string  formatted info message
   */
  std::string getInfoString() {

    if (is_saved_model_) {
      return getSavedModelInfoString(saved_model_);
    } else if (is_frozen_graph_) {
      return getGraphInfoString(graph_def_);
    } else {
      return "";
    }
  }

  /**
   * @brief Returns the underlying TensorFlow session.
   *
   * @return  tf::Session*  session
   */
  tf::Session* session() const {
    return session_;
  }

  /**
   * @brief Returns the underlying SavedModel.
   *
   * @return  const tf::SavedModelBundleLite&  SavedModel
   */
  const tf::SavedModelBundleLite& savedModel() const {
    return saved_model_;
  }

  /**
   * @brief Returns the underlying FrozenGraph GraphDef.
   *
   * @return  const tf::GraphDef& FrozenGraph GraphDef
   */
  const tf::GraphDef& frozenGraph() const {
    return graph_def_;
  }

  /**
   * @brief Returns whether loaded model is from SavedModel.
   *
   * @return  true   if loaded from SavedModel
   * @return  false  if not loaded from SavedModel
   */
  bool isSavedModel() const {
    return is_saved_model_;
  }

  /**
   * @brief Returns whether loaded model is from FrozenGraph.
   *
   * @return  true   if loaded from FrozenGraph
   * @return  false  if not loaded from FrozenGraph
   */
  bool isFrozenGraph() const {
    return is_frozen_graph_;
  }

  /**
   * @brief Returns number of model inputs.
   *
   * @return  int  number of inputs
   */
  int nInputs() const {
    return n_inputs_;
  }

  /**
   * @brief Returns number of model outputs.
   *
   * @return  int  number of outputs
   */
  int nOutputs() const {
    return n_outputs_;
  }

  /**
   * @brief Returns names of model inputs.
   *
   * @return  std::vector<std::string>  model input names
   */
  std::vector<std::string> inputNames() const {
    return input_names_;
  }

  /**
   * @brief Returns names of model outputs.
   *
   * @return  std::vector<std::string>  model output names
   */
  std::vector<std::string> outputNames() const {
    return output_names_;
  }

 protected:
  /**
   * @brief Runs the model once with dummy input to speed-up first inference.
   */
  void dummyCall() {

    // infer input shapes/types to create dummy input tensors
    auto input_shapes = getInputShapes();
    auto input_types = getInputTypes();
    std::vector<tf::Tensor> input_dummies;
    for (int k = 0; k < n_inputs_; k++) {
      std::vector<long int> dummy_shape(input_shapes[k].begin(),
                                        input_shapes[k].end());
      // Replace -1 (batch size dimension, None in python) with 1
      std::replace(dummy_shape.begin(), dummy_shape.end(), -1l, 1l);
      auto dummy_tensor_shape =
        tf::TensorShape(tf::gtl::ArraySlice<long int>(dummy_shape));
      tf::Tensor dummy(input_types[k], dummy_tensor_shape);
      // init to zero, based on type
      switch (input_types[k]) {
        case tf::DT_FLOAT:
          dummy.flat<float>().setZero();
          break;
        case tf::DT_DOUBLE:
          dummy.flat<double>().setZero();
        case tf::DT_INT32:
          dummy.flat<tf::int32>().setZero();
          break;
        case tf::DT_UINT32:
          dummy.flat<tf::uint32>().setZero();
          break;
        case tf::DT_UINT8:
          dummy.flat<tf::uint8>().setZero();
          break;
        case tf::DT_UINT16:
          dummy.flat<tf::uint16>().setZero();
          break;
        case tf::DT_INT16:
          dummy.flat<tf::int16>().setZero();
          break;
        case tf::DT_INT8:
          dummy.flat<tf::int8>().setZero();
          break;
        case tf::DT_STRING:
          dummy.flat<tf::tstring>().setZero();
          break;
        case tf::DT_COMPLEX64:
          dummy.flat<tf::complex64>().setZero();
          break;
        case tf::DT_COMPLEX128:
          dummy.flat<tf::complex128>().setZero();
          break;
        case tf::DT_INT64:
          dummy.flat<tf::int64>().setZero();
          break;
        case tf::DT_UINT64:
          dummy.flat<tf::uint64>().setZero();
          break;
        case tf::DT_BOOL:
          dummy.flat<bool>().setZero();
          break;
        case tf::DT_QINT8:
          dummy.flat<tf::qint8>().setZero();
          break;
        case tf::DT_QUINT8:
          dummy.flat<tf::quint8>().setZero();
          break;
        case tf::DT_QUINT16:
          dummy.flat<tf::quint16>().setZero();
          break;
        case tf::DT_QINT16:
          dummy.flat<tf::qint16>().setZero();
          break;
        case tf::DT_QINT32:
          dummy.flat<tf::qint32>().setZero();
          break;
        case tf::DT_BFLOAT16:
          dummy.flat<tf::bfloat16>().setZero();
          break;
        case tf::DT_HALF:
          dummy.flat<Eigen::half>().setZero();
          break;
      }
      input_dummies.push_back(dummy);
    }

    // run dummy inference
    volatile auto output_dummies = (*this)(input_dummies);
  }

 protected:
  /**
   * @brief underlying TensorFlow session
   */
  tf::Session* session_ = nullptr;

  /**
   * @brief underlying SavedModel
   */
  tf::SavedModelBundleLite saved_model_;

  /**
   * @brief underlying FrozenGraph GraphDef
   */
  tf::GraphDef graph_def_;

  /**
   * @brief whether loaded model is from SavedModel
   */
  bool is_saved_model_ = false;

  /**
   * @brief whether loaded model is from FrozenGraph
   */
  bool is_frozen_graph_ = false;

  /**
   * @brief number of model inputs
   */
  int n_inputs_;

  /**
   * @brief number of model outputs
   */
  int n_outputs_;

  /**
   * @brief (layer) names of model inputs
   */
  std::vector<std::string> input_names_;

  /**
   * @brief (layer) names of model outputs
   */
  std::vector<std::string> output_names_;

  /**
   * @brief mapping between SavedModel node and layer input/output names
   */
  std::unordered_map<std::string, std::string> saved_model_node2layer_;

  /**
   * @brief mapping between SavedModel layer and node input/output names
   */
  std::unordered_map<std::string, std::string> saved_model_layer2node_;
};


}  // namespace tensorflow_cpp
