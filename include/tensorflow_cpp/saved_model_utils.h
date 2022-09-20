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
 * @brief Utility functions for SavedModels
 */

#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow_cpp/utils.h>


namespace tensorflow_cpp {


namespace tf = tensorflow;


/**
 * @brief Loads a TensorFlow SavedModel from a directory into a new session.
 *
 * @param[in]  dir                              SavedModel directory
 * @param[in]  allow_growth                     dynamically grow GPU usage
 * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
 * @param[in]  visible_device_list              list of GPUs to use, e.g. "0,1"
 *
 * @return  tf::SavedModelBundleLite            SavedModel
 */
tf::SavedModelBundleLite loadSavedModel(
  const std::string& dir, const bool allow_growth = true,
  const double per_process_gpu_memory_fraction = 0,
  const std::string& visible_device_list = "") {

  tf::SavedModelBundleLite saved_model;
  tf::SessionOptions session_options = makeSessionOptions(
    allow_growth, per_process_gpu_memory_fraction, visible_device_list);
  tf::Status status =
    tf::LoadSavedModel(session_options, tf::RunOptions(), dir,
                       {tf::kSavedModelTagServe}, &saved_model);
  if (!status.ok())
    throw std::runtime_error("Failed to load SavedModel: " + status.ToString());

  return saved_model;
}


/**
 * @brief Loads a TensorFlow SavedModel from a directory into a new session.
 *
 * @param[in]  dir                              SavedModel directory
 * @param[in]  allow_growth                     dynamically grow GPU usage
 * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
 * @param[in]  visible_device_list              list of GPUs to use, e.g. "0,1"
 *
 * @return  tf::Session*                        session
 */
tf::Session* loadSavedModelIntoNewSession(
  const std::string& dir, const bool allow_growth = true,
  const double per_process_gpu_memory_fraction = 0,
  const std::string& visible_device_list = "") {

  tf::SavedModelBundleLite saved_model = loadSavedModel(
    dir, allow_growth, per_process_gpu_memory_fraction, visible_device_list);
  tf::Session* session = saved_model.GetSession();

  return session;
}


/**
 * @brief Returns the session that a SavedModel is loaded in.
 *
 * @param[in]  saved_model   SavedModel
 *
 * @return  tf::Session*  session
 */
tf::Session* getSessionFromSavedModel(
  const tf::SavedModelBundleLite& saved_model) {

  return saved_model.GetSession();
}


/**
 * @brief Determines the node name from a SavedModel layer name.
 *
 * Layer names are specified during model construction,
 * node names must be passed to `session->Run`.
 *
 * @param[in]  saved_model  SavedModel
 * @param[in]  layer_name   layer name
 * @param[in]  signature    SavedModel signature to query
 *
 * @return  std::string     node name
 */
std::string getSavedModelNodeByLayerName(
  const tf::SavedModelBundleLite& saved_model, const std::string& layer_name,
  const std::string& signature = "serving_default") {

  std::string node_name;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  auto inputs = model_def.inputs();
  auto outputs = model_def.outputs();
  auto& nodes = inputs;
  nodes.insert(outputs.begin(), outputs.end());
  for (const auto& node : nodes) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    if (key == layer_name) {
      node_name = info.name();
      break;
    }
  }

  return node_name;
}


/**
 * @brief Determines the layer name from a SavedModel node name.
 *
 * Layer names are specified during model construction,
 * node names must be passed to `session->Run`.
 *
 * @param[in]  saved_model  SavedModel
 * @param[in]  node_name    node name
 * @param[in]  signature    SavedModel signature to query
 *
 * @return  std::string     layer name
 */
std::string getSavedModelLayerByNodeName(
  const tf::SavedModelBundleLite& saved_model, const std::string& node_name,
  const std::string& signature = "serving_default") {

  std::string layer_name;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  auto inputs = model_def.inputs();
  auto outputs = model_def.outputs();
  auto& nodes = inputs;
  nodes.insert(outputs.begin(), outputs.end());
  for (const auto& node : nodes) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    if (info.name() == node_name) {
      layer_name = key;
      break;
    }
  }

  return layer_name;
}


/**
 * @brief Determines the names of the SavedModel input nodes.
 *
 * These are the names that need to be passed to `session->Run`.
 * Alternatively, using `layer_names`, the layer names can be returned.
 *
 * Returned names are sorted alphabetically, since their order is not
 * deterministic in general. The sorting is always based on the actual
 * node names, even when returning layer names.
 *
 * @param[in]  saved_model               SavedModel
 * @param[in]  layer_names               whether to return layer names
 * @param[in]  signature                 SavedModel signature to query
 *
 * @return  std::vector<std::string>     input names
 */
std::vector<std::string> getSavedModelInputNames(
  const tf::SavedModelBundleLite& saved_model, const bool layer_names = false,
  const std::string& signature = "serving_default") {

  std::vector<std::string> names;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  for (const auto& node : model_def.inputs()) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    names.push_back(info.name());
  }
  std::sort(names.begin(), names.end());

  if (layer_names) {
    std::vector<std::string> node_names = names;
    names = {};
    for (const auto& node_name : node_names)
      names.push_back(
        getSavedModelLayerByNodeName(saved_model, node_name, signature));
  }

  return names;
}


/**
 * @brief Determines the names of the SavedModel output nodes.
 *
 * These are the names that need to be passed to `session->Run`.
 * Alternatively, using `layer_names`, the layer names can be returned.
 *
 * Returned names are sorted alphabetically, since their order is not
 * deterministic in general. The sorting is always based on the actual
 * node names, even when returning layer names.
 *
 * @param[in]  saved_model               SavedModel
 * @param[in]  layer_names               whether to return layer names
 * @param[in]  signature                 SavedModel signature to query
 *
 * @return  std::vector<std::string>     output names
 */
std::vector<std::string> getSavedModelOutputNames(
  const tf::SavedModelBundleLite& saved_model, const bool layer_names = false,
  const std::string& signature = "serving_default") {

  std::vector<std::string> names;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  for (const auto& node : model_def.outputs()) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    names.push_back(info.name());
  }
  std::sort(names.begin(), names.end());

  if (layer_names) {
    std::vector<std::string> node_names = names;
    names = {};
    for (const auto& node_name : node_names)
      names.push_back(
        getSavedModelLayerByNodeName(saved_model, node_name, signature));
  }

  return names;
}


/**
 * @brief Determines the shape of a given SavedModel node.
 *
 * @param[in]  saved_model               SavedModel
 * @param[in]  node_name                 node name
 * @param[in]  signature                 SavedModel signature to query
 *
 * @return  std::vector<int>             node shape
 */
std::vector<int> getSavedModelNodeShape(
  const tf::SavedModelBundleLite& saved_model, const std::string& node_name,
  const std::string& signature = "serving_default") {

  std::vector<int> node_shape;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  auto inputs = model_def.inputs();
  auto outputs = model_def.outputs();
  auto& nodes = inputs;
  nodes.insert(outputs.begin(), outputs.end());
  for (const auto& node : nodes) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    if (info.name() == node_name) {
      const auto& shape = info.tensor_shape();
      for (int d = 0; d < shape.dim_size(); d++)
        node_shape.push_back(shape.dim(d).size());
      break;
    }
  }

  return node_shape;
}


/**
 * @brief Determines the datatype of a given SavedModel node.
 *
 * @param[in]  saved_model   SavedModel
 * @param[in]  node_name     node name
 * @param[in]  signature     SavedModel signature to query
 *
 * @return  tf::DataType     node datatype
 */
tf::DataType getSavedModelNodeType(
  const tf::SavedModelBundleLite& saved_model, const std::string& node_name,
  const std::string& signature = "serving_default") {

  tf::DataType type = tf::DT_INVALID;
  const tf::SignatureDef& model_def = saved_model.GetSignatures().at(signature);
  auto inputs = model_def.inputs();
  auto outputs = model_def.outputs();
  auto& nodes = inputs;
  nodes.insert(outputs.begin(), outputs.end());
  for (const auto& node : nodes) {
    const std::string& key = node.first;
    const tf::TensorInfo& info = node.second;
    if (info.name() == node_name) {
      type = info.dtype();
      break;
    }
  }

  return type;
}


/**
 * Returns information about a SavedModel model.
 *
 * Returns a formatted message containing information about the shape and type
 * of all inputs/outputs of all SavedModel signatures.
 *
 * @param[in]  saved_model  SavedModel
 *
 * @return  std::string   formatted info message
 */
std::string getSavedModelInfoString(
  const tf::SavedModelBundleLite& saved_model) {

  std::stringstream ss;
  ss << "SavedModel Info:" << std::endl;

  ss << "Signatures:" << std::endl;
  const auto& signatures = saved_model.GetSignatures();
  for (const auto& sig : signatures) {

    ss << "  " << sig.first << std::endl;
    const auto& def = sig.second;

    ss << "    Inputs: " << def.inputs_size() << std::endl;
    for (const auto& node : def.inputs()) {
      ss << "      " << node.first << ": " << node.second.name() << std::endl;
      ss << "        Shape: [ ";
      for (int d = 0; d < node.second.tensor_shape().dim_size(); d++) {
        ss << node.second.tensor_shape().dim(d).size() << ", ";
      }
      ss << "]" << std::endl;
      ss << "        DataType: " << tf::DataTypeString(node.second.dtype())
         << std::endl;
    }

    ss << "    Outputs: " << def.outputs_size() << std::endl;
    for (const auto& node : def.outputs()) {
      ss << "      " << node.first << ": " << node.second.name() << std::endl;
      ss << "        Shape: [ ";
      for (int d = 0; d < node.second.tensor_shape().dim_size(); d++) {
        ss << node.second.tensor_shape().dim(d).size() << ", ";
      }
      ss << "]" << std::endl;
      ss << "        DataType: " << tf::DataTypeString(node.second.dtype())
         << std::endl;
    }
  }

  return ss.str();
}


}  // namespace tensorflow_cpp