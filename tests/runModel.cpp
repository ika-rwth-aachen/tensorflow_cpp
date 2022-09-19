#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow_cpp/model.h>


std::string model_path;
std::string img_path;
int actual_digit;


int main(int argc, char** argv) {

  ::testing::InitGoogleTest(&argc, argv);
  model_path = argv[1];
  img_path = argv[2];
  actual_digit = std::stoi(img_path.substr(img_path.size() - 5, 1));
  return RUN_ALL_TESTS();
}


TEST(tensorflow_cpp, runModel) {

  // define graph for loading input image (pure TensorFlow C++)
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session(scope);
  auto read_file_op = tensorflow::ops::ReadFile(scope, img_path);
  auto decode_jpeg_op = tensorflow::ops::DecodeJpeg(scope, read_file_op);
  auto cast_op = tensorflow::ops::Cast(scope, decode_jpeg_op, tensorflow::DT_FLOAT);
  auto const_op = tensorflow::ops::Const(scope, {float(255.0)});
  auto div_op = tensorflow::ops::Div(scope, cast_op, const_op);

  // execute graph to load input tensor (pure TensorFlow C++)
  std::vector<tensorflow::Tensor> outputs;
  session.Run({div_op}, &outputs);
  tensorflow::Tensor input_tensor = outputs[0];

  // load and run model (tensorflow_cpp)
  tensorflow_cpp::Model model;
  model.loadModel(model_path);
  auto out = model(input_tensor);

  // find most likely prediction and print probabilities
  int predicted_digit = 0;
  float max_probability = 0.0;
  std::cout << "Digit | Probability" << std::fixed << std::setprecision(2) << std::endl;
  for (int i = 0; i < out.shape().dim_size(1); i++) {
    float probability = out.tensor<float, 2>()(0, i);
    std::cout << "   " << i << "  |  " << (probability * 100) << "%" << std::endl;
    if (probability > max_probability) {
      max_probability = probability;
      predicted_digit = i;
    }
  }

  // test prediction
  EXPECT_EQ(predicted_digit, actual_digit);
}
