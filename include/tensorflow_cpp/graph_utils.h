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
 * @brief Utility functions for FrozenGraphs
 */

#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensorflow/core/platform/env.h>
#include <tensorflow_cpp/utils.h>


namespace tensorflow_cpp {


namespace tf = tensorflow;


/**
 * @brief Loads a TensorFlow graph from a frozen graph file.
 *
 * @param[in]  file          frozen graph file
 *
 * @return  tf::GraphDef     graph
 */
inline tf::GraphDef loadFrozenGraph(const std::string& file) {

  tf::GraphDef graph_def;
  tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), file, &graph_def);
  if (!status.ok())
    throw std::runtime_error("Failed to load frozen graph: " +
                             status.ToString());

  return graph_def;
}


/**
 * @brief Loads a TensorFlow graph into an existing session.
 *
 * @param[in]  session      session
 * @param[in]  graph_def    graph
 *
 * @return  true            if operation succeeded
 * @return  false           if operation failed
 */
inline bool loadGraphIntoSession(tf::Session* session,
                                 const tf::GraphDef& graph_def) {

  tf::Status status = session->Create(graph_def);
  if (!status.ok())
    throw std::runtime_error("Failed to load graph into session: " +
                             status.ToString());

  return true;
}

/**
 * @brief Loads a TensorFlow graph from a frozen graph file into a new
 * session.
 *
 * @param[in]  file                             frozen graph file
 * @param[in]  allow_growth                     dynamically grow GPU usage
 * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
 * @param[in]  visible_device_list              list of GPUs to use, e.g.
 * "0,1"
 *
 * @return  tf::Session*                        session
 */
inline tf::Session* loadFrozenGraphIntoNewSession(
  const std::string& file, const bool allow_growth = true,
  const double per_process_gpu_memory_fraction = 0,
  const std::string& visible_device_list = "") {

  tf::GraphDef graph_def = loadFrozenGraph(file);
  tf::Session* session = createSession(
    allow_growth, per_process_gpu_memory_fraction, visible_device_list);
  if (!loadGraphIntoSession(session, graph_def)) return nullptr;

  return session;
}


/**
 * @brief Determines the names of all graph input nodes.
 *
 * @param[in]  graph_def                 graph
 *
 * @return  std::vector<std::string>     list of input node names
 */
inline std::vector<std::string> getGraphInputNames(
  const tf::GraphDef& graph_def) {

  std::vector<std::string> input_nodes;
  for (const tf::NodeDef& node : graph_def.node()) {
    if (node.op() == "Placeholder") input_nodes.push_back(node.name());
  }

  return input_nodes;
}


/**
 * @brief Determines the names of all graph output nodes.
 *
 * @param[in]  graph_def                 graph
 *
 * @return  std::vector<std::string>     list of output node names
 */
inline std::vector<std::string> getGraphOutputNames(
  const tf::GraphDef& graph_def) {

  std::vector<std::string> output_nodes;
  std::vector<std::string> nodes_with_outputs;
  std::unordered_set<std::string> unlikely_output_ops = {"Const", "Assign",
                                                         "NoOp", "Placeholder",
                                                         "Assert"};
  for (const tf::NodeDef& node : graph_def.node()) {
    for (const std::string& input_name : node.input())
      nodes_with_outputs.push_back(input_name);
  }
  for (const tf::NodeDef& node : graph_def.node()) {
    if (std::find(nodes_with_outputs.begin(), nodes_with_outputs.end(),
                  node.name()) == nodes_with_outputs.end() &&
        unlikely_output_ops.count(node.op()) == 0)
      output_nodes.push_back(node.name());
  }

  return output_nodes;
}


/**
 * @brief Determines the shape of a given graph node.
 *
 * @param[in]  graph_def         graph
 * @param[in]  node_name         node name
 *
 * @return  std::vector<int>     node shape
 */
inline std::vector<int> getGraphNodeShape(const tf::GraphDef& graph_def,
                                          const std::string& node_name) {

  std::vector<int> node_shape;
  for (const tf::NodeDef& node : graph_def.node()) {
    if (node.name() == node_name) {
      if (node.attr().count("shape") == 0) return node_shape;
      auto shape = node.attr().at("shape").shape();
      for (int d = 0; d < shape.dim_size(); d++)
        node_shape.push_back(shape.dim(d).size());
      break;
    }
  }

  return node_shape;
}


/**
 * @brief Determines the datatype of a given graph node.
 *
 * @param[in]  graph_def     graph
 * @param[in]  node_name     node name
 *
 * @return  tf::DataType     node datatype
 */
inline tf::DataType getGraphNodeType(const tf::GraphDef& graph_def,
                                     const std::string& node_name) {

  tf::DataType type = tf::DT_INVALID;
  for (const tf::NodeDef& node : graph_def.node()) {
    if (node.name() == node_name) {
      if (node.attr().count("dtype") == 0) return type;
      type = node.attr().at("dtype").type();
      break;
    }
  }
  return type;
}


/**
 * Returns information about a FrozenGraph model.
 *
 * Returns a formatted message containing information about the shape and type
 * of all inputs/outputs of a FrozenGraph.
 *
 * Currently limited to single-output graphs.
 *
 * @param[in]  graph_def  graph
 *
 * @return  std::string   formatted info message
 */
inline std::string getGraphInfoString(const tf::GraphDef& graph_def) {

  std::stringstream ss;
  ss << "FrozenGraph Info:" << std::endl;

  const std::vector<std::string> inputs = getGraphInputNames(graph_def);
  const std::vector<std::string> outputs = getGraphOutputNames(graph_def);

  ss << "Inputs: " << inputs.size() << std::endl;
  for (const auto& name : inputs) {
    const auto& shape = getGraphNodeShape(graph_def, name);
    const auto& dtype = getGraphNodeType(graph_def, name);
    ss << "  " << name << std::endl;
    ss << "    Shape: [ ";
    for (int d = 0; d < shape.size(); d++) {
      ss << shape[d] << ", ";
    }
    ss << "]" << std::endl;
    ss << "    DataType: " << tf::DataTypeString(dtype) << std::endl;
  }

  ss << "Outputs: " << outputs.size() << std::endl;
  for (const auto& name : outputs) {
    const auto& shape = getGraphNodeShape(graph_def, name);
    const auto& dtype = getGraphNodeType(graph_def, name);
    ss << "  " << name << std::endl;
    ss << "    Shape: [ ";
    for (int d = 0; d < shape.size(); d++) {
      ss << shape[d] << ", ";
    }
    ss << "]" << std::endl;
    ss << "    DataType: " << tf::DataTypeString(dtype) << std::endl;
  }

  return ss.str();
}


}  // namespace tensorflow_cpp
