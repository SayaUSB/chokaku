cmake_minimum_required(VERSION 3.16)
project(chokaku VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenVINO REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(CSV_LIB csv)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# Executable
add_executable(chokaku 
    src/chokaku/chokaku_main.cpp 
    src/chokaku/chokaku.cpp
    src/chokaku/utils/config.cpp
)

# Test executable
add_executable(chokaku_test
    test/chokaku_test.cpp
    src/chokaku/chokaku.cpp
    src/chokaku/utils/config.cpp
)

target_include_directories(chokaku PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_include_directories(chokaku_test PRIVATE 
    ${PORTAUDIO_INCLUDE_DIRS}
    include
)

target_link_libraries(chokaku 
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_link_libraries(chokaku_test 
    openvino::runtime
    ${PORTAUDIO_LIBRARIES}
    ${CSV_LIB}
)

target_compile_options(chokaku PRIVATE 
    ${PORTAUDIO_CFLAGS_OTHER}
    -Wall -Wextra -O3
)

# Install
install(TARGETS chokaku chokaku_test DESTINATION bin)