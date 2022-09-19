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


TEST(tensorflow_cpp, getTypes) {

  tensorflow_cpp::Model model(model_path);

  EXPECT_EQ(model.getInputType(), tensorflow::DT_FLOAT);
  EXPECT_EQ(model.getOutputType(), tensorflow::DT_FLOAT);
}
