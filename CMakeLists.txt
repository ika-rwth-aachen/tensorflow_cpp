cmake_minimum_required(VERSION 3.8)
project(tensorflow_cpp VERSION 1.0.6 LANGUAGES CXX)

find_package(ros_environment QUIET)
if(ros_environment_FOUND)
  set(ROS_VERSION $ENV{ROS_VERSION})
else()
  set(ROS_VERSION 0)
endif()

find_package(TensorFlow REQUIRED)

# === ROS2 / AMENT =============================================================
if(${ROS_VERSION} EQUAL 2)

  # find dependencies
  find_package(ament_cmake REQUIRED)

  add_library(tensorflow_cpp INTERFACE)

  target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TensorFlow_INCLUDE_DIRS}
  )
  target_link_libraries(${PROJECT_NAME} INTERFACE
    ${TensorFlow_LIBRARIES}
  )

  ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
  ament_export_dependencies(TensorFlow)

  install(
    DIRECTORY include/
    DESTINATION include
  )

  install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
  endif()

  ament_package()

# === ROS1 (CATKIN) ============================================================
elseif(${ROS_VERSION} EQUAL 1)

  find_package(catkin REQUIRED)

  catkin_package(
    INCLUDE_DIRS include
    DEPENDS TensorFlow
  )

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
  )

  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
  )

# === STANDARD CMAKE ===========================================================
# https://stackoverflow.com/a/49143782
else()

  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TensorFlow_INCLUDE_DIRS}
  )
  target_link_libraries(${PROJECT_NAME} INTERFACE
    ${TensorFlow_LIBRARIES}
  )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 1.0.1
    COMPATIBILITY AnyNewerVersion
  )

  install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
  )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  )

  install(EXPORT ${PROJECT_NAME}Targets DESTINATION lib/cmake/${PROJECT_NAME})
  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
                "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
          DESTINATION lib/cmake/${PROJECT_NAME})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

  option(BUILD_TESTING "" OFF)
  include(CTest)
  if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR ${PROJECT_NAME}_BUILD_TESTING) AND BUILD_TESTING)
    add_subdirectory(tests)
  endif()

endif()
