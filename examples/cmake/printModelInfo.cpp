#include <iostream>
#include <string>

#include <tensorflow_cpp/model.h>


int main(int argc, char **argv) {

  tensorflow_cpp::Model model;
  if (argc > 1) model.loadModel(argv[1]);
  std::cout << model.getInfoString() << std::endl;
}
