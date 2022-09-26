include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(SavedModelPath ${PROJECT_SOURCE_DIR}/examples/models/saved_model)
set(FrozenGraphPath ${PROJECT_SOURCE_DIR}/examples/models/frozen_graph/frozen_graph.pb)
set(MnistPath ${PROJECT_SOURCE_DIR}/examples/mnist)

add_executable(loadModel loadModel.cpp)
add_executable(printModelInfo printModelInfo.cpp)
add_executable(getShapes getShapes.cpp)
add_executable(getTypes getTypes.cpp)
add_executable(runModel runModel.cpp)

target_link_libraries(loadModel PRIVATE tensorflow_cpp GTest::gtest_main)
target_link_libraries(printModelInfo PRIVATE tensorflow_cpp GTest::gtest_main)
target_link_libraries(getShapes PRIVATE tensorflow_cpp GTest::gtest_main)
target_link_libraries(getTypes PRIVATE tensorflow_cpp GTest::gtest_main)
target_link_libraries(runModel PRIVATE tensorflow_cpp GTest::gtest_main)

add_test(NAME test_loadModel_SavedModel  COMMAND loadModel ${SavedModelPath})
add_test(NAME test_loadModel_FrozenGraph COMMAND loadModel ${FrozenGraphPath})

add_test(NAME test_printModelInfo_SavedModel  COMMAND loadModel ${SavedModelPath})
add_test(NAME test_printModelInfo_FrozenGraph COMMAND loadModel ${FrozenGraphPath})

add_test(NAME test_getShapes_SavedModel  COMMAND getShapes ${SavedModelPath})
# add_test(NAME test_getShapes_FrozenGraph COMMAND getShapes ${FrozenGraphPath})

add_test(NAME test_getTypes_SavedModel  COMMAND getTypes ${SavedModelPath})
# add_test(NAME test_getTypes_FrozenGraph COMMAND getTypes ${FrozenGraphPath})

add_test(NAME test_runModel_0_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/0.jpg)
add_test(NAME test_runModel_1_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/1.jpg)
add_test(NAME test_runModel_2_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/2.jpg)
add_test(NAME test_runModel_3_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/3.jpg)
add_test(NAME test_runModel_4_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/4.jpg)
add_test(NAME test_runModel_5_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/5.jpg)
add_test(NAME test_runModel_6_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/6.jpg)
add_test(NAME test_runModel_7_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/7.jpg)
add_test(NAME test_runModel_8_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/8.jpg)
add_test(NAME test_runModel_9_SavedModel  COMMAND runModel ${SavedModelPath} ${MnistPath}/9.jpg)
add_test(NAME test_runModel_0_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/0.jpg)
add_test(NAME test_runModel_1_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/1.jpg)
add_test(NAME test_runModel_2_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/2.jpg)
add_test(NAME test_runModel_3_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/3.jpg)
add_test(NAME test_runModel_4_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/4.jpg)
add_test(NAME test_runModel_5_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/5.jpg)
add_test(NAME test_runModel_6_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/6.jpg)
add_test(NAME test_runModel_7_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/7.jpg)
add_test(NAME test_runModel_8_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/8.jpg)
add_test(NAME test_runModel_9_FrozenGraph COMMAND runModel ${SavedModelPath} ${MnistPath}/9.jpg)