#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tensorflow_cpp/model.h>


std::string model_path;


int main(int argc, char** argv) {

  ::testing::InitGoogleTest(&argc, argv);
  model_path = argv[1];
  return RUN_ALL_TESTS();
}


TEST(tensorflow_cpp, getShapes) {

  tensorflow_cpp::Model model(model_path);

  // test input shape
  std::vector<int> expected_input_shape = {-1, 28, 28};
  std::vector<int> input_shape = model.getInputShape();
  ASSERT_EQ(input_shape.size(), expected_input_shape.size());
  for (int d = 0; d < input_shape.size(); d++)
    EXPECT_EQ(input_shape[d], expected_input_shape[d]);

  // test output shape
  std::vector<int> expected_output_shape = {-1, 10};
  auto output_shape = model.getOutputShape();
  ASSERT_EQ(output_shape.size(), expected_output_shape.size());
  for (int d = 0; d < output_shape.size(); d++)
    EXPECT_EQ(output_shape[d], expected_output_shape[d]);
}
