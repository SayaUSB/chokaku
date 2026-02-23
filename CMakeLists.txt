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

add_executable(main 
    src/chokaku/chokaku_main.cpp 
    src/chokaku/node/chokaku_node.cpp
    src/chokaku/inference/chokaku_inference.cpp
    src/chokaku/utils/config.cpp
    src/chokaku/utils/prediction.cpp
    src/chokaku/utils/audio_processing.cpp
    src/chokaku/utils/audio_capture.cpp
)

add_executable(check
    test/chokaku_test.cpp
    src/chokaku/inference/chokaku_inference.cpp
    src/chokaku/utils/config.cpp
    src/chokaku/utils/prediction.cpp
    src/chokaku/utils/audio_processing.cpp
    src/chokaku/utils/audio_capture.cpp
)

ament_target_dependencies(main rclcpp std_msgs)

target_include_directories(main PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_include_directories(check PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_link_libraries(main 
    rclcpp::rclcpp
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_link_libraries(check 
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_compile_options(main PRIVATE 
    ${PORTAUDIO_CFLAGS_OTHER}
    -Wall -Wextra -O3
)

target_compile_options(check PRIVATE 
    ${PORTAUDIO_CFLAGS_OTHER}
    -Wall -Wextra -O3
)

install(TARGETS main check
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