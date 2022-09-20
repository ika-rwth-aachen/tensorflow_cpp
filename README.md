# tensorflow_cpp

<p align="center">
  <img src="https://img.shields.io/github/v/release/ika-rwth-aachen/tensorflow_cpp"/>
  <img src="https://img.shields.io/github/license/ika-rwth-aachen/tensorflow_cpp"/>
  <a href="https://ika-rwth-aachen.github.io/tensorflow_cpp"><img src="https://github.com/ika-rwth-aachen/tensorflow_cpp/actions/workflows/doc.yml/badge.svg"/></a>
  <img src="https://img.shields.io/badge/ROS1-noetic-green"/>
  <img src="https://img.shields.io/github/stars/ika-rwth-aachen/tensorflow_cpp?style=social"/>
</p>

*tensorflow_cpp* is a header-only library that provides helpful wrappers around the [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc), allowing you to **easily load, inspect, and run saved models and frozen graphs in C++**. The library is easy to integrate into CMake projects, but is also available as a [ROS](https://www.ros.org/) package.

- [Examples](#examples)
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [CMake](#cmake)
  - [ROS](#ros)
- [Testing](#testing)
- [Documentation](#documentation)


## Examples

*Loading and running a single-input/single-output model*

```cpp
#include <iostream>
#include <string>
#include <vector>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow_cpp/model.h>

// load single-input/single-output model
std::string model_dir = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_dir);

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
std::string model_dir = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_dir);

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
std::string model_dir = "/PATH/TO/MODEL";
tensorflow_cpp::Model model;
model.loadModel(model_dir);

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

*tensorflow_cpp* is a wrapper around the official [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc). The C++ API including `libtensorflow_cc.so` must be installed on the system. Unfortunately, TensorFlow does not provide proper instructions for building the C++ API from source.

> **Note**
> We hope to soon release a pre-built TensorFlow C++ API package for easy installation.

In the meantime, you can try your luck at building the TensorFlow C++ API from source yourself. Once the library and headers are installed, you can use the provided [`TensorFlowConfig.cmake`](cmake/TensorFlowConfig.cmake) to enable the `find_package(TensorFlow REQUIRED)` call in *tensorflow_cpp's* [`CMakeLists.txt`](CMakeLists.txt).

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
    include_directories(foo ${tensorflow_cpp_INCLUDE_DIRS})
    target_link_libraries(foo PRIVATE ${tensorflow_cpp_LIBRARIES})
    ```

### ROS

1. Clone this repository into your ROS workspace.

    ```bash
    git clone https://github.com/ika-rwth-aachen/tensorflow_cpp.git
    cd tensorflow_cpp
    ```

1. In order to include *tensorflow_cpp* in a ROS package, specify the dependency in its `package.xml` and use `find_package()` in your package's `CMakeLists.txt`.

    ```xml
    <!-- package.xml -->
    <depend>tensorflow_cpp</depend>
    ```

    ```cmake
    # CMakeLists.txt
    find_package(catkin REQUIRED COMPONENTS
      tensorflow_cpp
    )
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
# tensorflow_cpp$
doxygen doc/Doxyfile
```
