cmake_minimum_required(VERSION 3.26)
project(gpgpu_raytrace_rainbow CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(gpgpu_raytrace_rainbow rainbow.cu stb_image_write.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set_target_properties(gpgpu_raytrace_rainbow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)