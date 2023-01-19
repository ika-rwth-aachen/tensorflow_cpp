# tensorflow_cpp

<p align="center">
  <img src="https://img.shields.io/github/v/release/ika-rwth-aachen/tensorflow_cpp"/>
  <img src="https://img.shields.io/github/license/ika-rwth-aachen/tensorflow_cpp"/>
  <a href="https://zenodo.org/badge/latestdoi/537518212"><img src="https://zenodo.org/badge/537518212.svg"></a>
  <a href="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/build.yml"><img src="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/build.yml/badge.svg"/></a>
  <a href="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/test.yml"><img src="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/test.yml/badge.svg"/></a>
  <a href="https://ika-rwth-aachen.github.io/tensorflow_cpp"><img src="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/doc.yml/badge.svg"/></a>
  <img src="https://img.shields.io/badge/ROS1-noetic-green"/>
  <img src="https://img.shields.io/badge/ROS2-humble-green"/>
  <a href="https://github.com/ika-rwth-aachen/tensorflow_cpp"><img src="https://img.shields.io/github/stars/ika-rwth-aachen/tensorflow_cpp?style=social"/></a>
</p>

*tensorflow_cpp* is a header-only library that provides helpful wrappers around the [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc), allowing you to **easily load, inspect, and run saved models and frozen graphs in C++**. The library is easy to integrate into CMake projects, but is also available as a [ROS and ROS2](https://www.ros.org/) package.

If you want to use the TensorFlow C++ API to load, inspect, and run saved models and frozen graphs in C++, we suggest that you also check out our helper library tensorflow_cpp. 

If you are looking for an easy way to install the TensorFlow C++ API, we suggest that you also check out our repository [*libtensorflow_cc*](https://github.com/ika-rwth-aachen/libtensorflow_cc). There, we provide a pre-built library and a Docker image for easy installation and usage of the TensorFlow C++ API. <a href="https://github.com/ika-rwth-aachen/libtensorflow_cc"><img src="https://img.shields.io/github/stars/ika-rwth-aachen/libtensorflow_cc?style=social"/></a>

---

- [tensorflow\_cpp](#tensorflow_cpp)
  - [Examples](#examples)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [CMake](#cmake)
    - [ROS/ROS2](#rosros2)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Acknowledgements](#acknowledgements)
  - [Notice](#notice)


## Examples

*Loading and running a single-input/single-output model*

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow_cpp/model.h>

// load single-input/single-output model
std::string model_path = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_path);

// log model info
std::cout << model.getInfoString() << std::endl;

// get input/output shape/type, if required
std::vector<int> input_shape = model.getInputShape();
tensorflow::DataType output_type = model.getOutputType();
// ... do something ...

// create and fill input tensor
tensorflow::Tensor input_tensor;
// ... fill input tensor ...

// run model
tensorflow::Tensor output_tensor = model(input_tensor);
```

<details>
<summary><i>Loading and running a multi-input/multi-output model</i></summary>

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow_cpp/model.h>

// load multi-input/multi-output model
std::string model_path = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_path);

// log model info
std::cout << model.getInfoString() << std::endl;

// input/output layer names are determined automatically,
// but could potentially have different order than expected

// get input/output shapes/types, if required
std::vector<int> input_shape_1 = model.getNodeShapes()[0];
tensorflow::DataType output_type_2 = model.getNodeTypes()[1];
// ... do something ...

// create and fill input tensors
tensorflow::Tensor input_tensor_1;
tensorflow::Tensor input_tensor_2;
// ... fill input tensors ...

// run model
auto outputs = model({input_tensor_1, input_tensor_2});
tensorflow::Tensor output_tensor_1& = outputs[0];
tensorflow::Tensor output_tensor_2& = outputs[1];
```

</details>

<details>
<summary><i>Loading and running a multi-input/multi-output model with specific inputs/outputs</i></summary>

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow_cpp/model.h>

// load multi-input/multi-output model
std::string model_path = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_path);

// log model info
std::cout << model.getInfoString() << std::endl;

// set model input/output layer names (see `model.logInfo()`)
const std::string kModelInputName1 = "input1";
const std::string kModelInputName2 = "input2";
const std::string kModelOutputName1 = "output1";
const std::string kModelOutputName2 = "output2";

// get input/output shapes/types, if required
std::vector<int> input_shape_1 = model.getNodeShape(kModelInputName1);
tensorflow::DataType output_type_2 = model.getNodeType(kModelOutputName2);
// ... do something ...

// create and fill input tensors
tensorflow::Tensor input_tensor_1;
tensorflow::Tensor input_tensor_2;
// ... fill input tensors ...

// run model
auto outputs = model({{kModelInputName1, input_tensor_1}, {kModelInputName2, input_tensor_2}}, {kModelOutputName1, kModelOutputName2});
tensorflow::Tensor output_tensor_1& = outputs[kModelOutputName1];
tensorflow::Tensor output_tensor_2& = outputs[kModelOutputName2];
```

</details>


## Installation

### Dependencies

*tensorflow_cpp* is a wrapper around the official [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc). The C++ API including `libtensorflow_cc.so` must be installed on the system.

Instead of having to build the C++ API from source yourself, we recommend to check out our repository [*libtensorflow_cc*](https://github.com/ika-rwth-aachen/libtensorflow_cc). There, we provide a pre-built library and a Docker image for easy installation and usage of the TensorFlow C++ API.

Installation is as easy as the following. Head over to [*libtensorflow_cc*](https://github.com/ika-rwth-aachen/libtensorflow_cc) for more details.
```
ARCH=$(dpkg --print-architecture)
wget https://github.com/ika-rwth-aachen/libtensorflow_cc/releases/download/v2.9.2/libtensorflow-cc_2.9.2-gpu_${ARCH}.deb
sudo dpkg -i libtensorflow-cc_2.9.2-gpu_${ARCH}.deb
ldconfig
```

If you have already installed the C++ API another way, you can use the provided [`TensorFlowConfig.cmake`](cmake/TensorFlowConfig.cmake) to enable the `find_package(TensorFlow REQUIRED)` call in *tensorflow_cpp's* [`CMakeLists.txt`](CMakeLists.txt).

### CMake

1. Clone this repository.

    ```bash
    git clone https://github.com/ika-rwth-aachen/tensorflow_cpp.git
    cd tensorflow_cpp
    ```

2. Install *tensorflow_cpp* system-wide.

    ```bash
    # tensorflow_cpp$
    mkdir -p build
    cd build
    cmake ..
    sudo make install
    ```

3. Use `find_package()` to locate and integrate *tensorflow_cpp* into your CMake project. See the [CMake example project](examples/cmake/).

    ```cmake
    # CMakeLists.txt
    find_package(tensorflow_cpp REQUIRED)
    # ...
    add_executable(foo ...) # / add_library(foo ...)
    # ...
    target_link_libraries(foo tensorflow_cpp)
    ```

### ROS/ROS2

1. Clone this repository into your ROS/ROS2 workspace.

    ```bash
    git clone https://github.com/ika-rwth-aachen/tensorflow_cpp.git
    cd tensorflow_cpp
    ```

1. In order to include *tensorflow_cpp* in a ROS/ROS2 package, specify the dependency in its `package.xml` and use `find_package()` in your package's `CMakeLists.txt`.

    ```xml
    <!-- package.xml -->
    <depend>tensorflow_cpp</depend>
    ```

    ```cmake
    # CMakeLists.txt

    # ROS
    find_package(catkin REQUIRED COMPONENTS
      tensorflow_cpp
    )

    # ROS2
    find_package(tensorflow_cpp REQUIRED)
    ament_target_dependencies(<TARGET> tensorflow_cpp)
    ```


## Testing

In order to build and run the test cases defined in [`tests/`](tests/), execute the following.

```bash
# tensorflow_cpp$
mkdir -p build
cd build
cmake -DBUILD_TESTING=ON ..
make
ctest
```


## Documentation

[Click here](https://ika-rwth-aachen.github.io/tensorflow_cpp) to be taken to the full API documentation.

The documentation can be generated by running [Doxygen](https://doxygen.nl/).

```bash
# tensorflow_cpp/doc$
doxygen
```


## Acknowledgements

This work is accomplished within the projects [6GEM](https://6gem.de/) (FKZ 16KISK038) and [UNICAR*agil*](https://www.unicaragil.de/) (FKZ 16EMO0284K). We acknowledge the financial support for the projects by the Federal Ministry of Education and Research of Germany (BMBF).


## Notice

This repository is not endorsed by or otherwise affiliated with [TensorFlow](https://www.tensorflow.org) or Google. TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc. [TensorFlow](https://github.com/tensorflow/tensorflow) is released under the [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).