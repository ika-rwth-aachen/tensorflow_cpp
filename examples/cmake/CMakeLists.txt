cmake_minimum_required(VERSION 3.0.2)
project(printModelInfo VERSION 1.0.0 LANGUAGES CXX)

find_package(tensorflow_cpp REQUIRED)

add_executable(${PROJECT_NAME} printModelInfo.cpp)

include_directories(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} tensorflow_cpp)
