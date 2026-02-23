cmake_minimum_required(VERSION 3.16)
project(chokaku VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(CSV_LIB csv)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

add_executable(chokaku_node 
    src/chokaku/chokaku_main.cpp 
    src/chokaku/node/chokaku_node.cpp
    src/chokaku/inference/chokaku_inference.cpp
    src/chokaku/utils/config.cpp
    src/chokaku/utils/prediction.cpp
    src/chokaku/utils/audio_processing.cpp
    src/chokaku/utils/audio_capture.cpp
)

add_executable(chokaku_test
    test/chokaku_test.cpp
    src/chokaku/inference/chokaku_inference.cpp
    src/chokaku/utils/config.cpp
    src/chokaku/utils/prediction.cpp
    src/chokaku/utils/audio_processing.cpp
    src/chokaku/utils/audio_capture.cpp
)

ament_target_dependencies(chokaku_node rclcpp std_msgs)

target_include_directories(chokaku_node PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_include_directories(chokaku_test PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_link_libraries(chokaku_node 
    rclcpp::rclcpp
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_link_libraries(chokaku_test 
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_compile_options(chokaku_node PRIVATE 
    ${PORTAUDIO_CFLAGS_OTHER}
    -Wall -Wextra -O3
)

target_compile_options(chokaku_test PRIVATE 
    ${PORTAUDIO_CFLAGS_OTHER}
    -Wall -Wextra -O3
)

install(TARGETS chokaku_node chokaku_test
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# Install model files
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()