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


#pragma once

#include <stdexcept>
#include <string>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>


namespace tensorflow_cpp {


namespace tf = tensorflow;


/**
 * @brief Helps to quickly create SessionOptions.
 *
 * @param[in]  allow_growth                     dynamically grow GPU usage
 * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
 * @param[in]  visible_device_list              list of GPUs to use, e.g. "0,1"
 *
 * @return  tf::SessionOptions                  session options
 */
tf::SessionOptions makeSessionOptions(
  const bool allow_growth = true,
  const double per_process_gpu_memory_fraction = 0,
  const std::string& visible_device_list = "") {

  tf::SessionOptions options = tf::SessionOptions();
  tf::ConfigProto* config = &options.config;
  tf::GPUOptions* gpu_options = config->mutable_gpu_options();
  gpu_options->set_allow_growth(allow_growth);
  gpu_options->set_per_process_gpu_memory_fraction(
    per_process_gpu_memory_fraction);
  gpu_options->set_visible_device_list(visible_device_list);

  return options;
}


/**
 * @brief Creates a new TensorFlow session.
 *
 * @param[in]  allow_growth                     dynamically grow GPU usage
 * @param[in]  per_process_gpu_memory_fraction  maximum GPU memory fraction
 * @param[in]  visible_device_list              list of GPUs to use, e.g. "0,1"
 *
 * @return  tf::Session*                        session
 */
tf::Session* createSession(const bool allow_growth = true,
                           const double per_process_gpu_memory_fraction = 0,
                           const std::string& visible_device_list = "") {

  tf::Session* session;
  tf::SessionOptions options = makeSessionOptions(
    allow_growth, per_process_gpu_memory_fraction, visible_device_list);
  tf::Status status = tf::NewSession(options, &session);
  if (!status.ok())
    throw std::runtime_error("Failed to create new session: " +
                             status.ToString());

  return session;
}


}  // namespace tensorflow_cpp