cmake_minimum_required(VERSION 3.26)
project(gpgpu_raytrace_rainbow)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(gpgpu_raytrace_rainbow rainbow.cpp)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -Wall -Wextra")