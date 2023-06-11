# Raytrace the rainbow on the GPU

My solution for the the [Raytrace the rainbow on the GPU](https://docs.google.com/document/d/15x38bYtuHRw_0vjiJDCSkuslPVO1VjPjjNfWkqnX-rE/edit#heading=h.xo7isdmcpzd8) assignment, which is part of the GPGPU course at Eötvös Loránd University.

## Prerequisities

The following dependencies are required:

- CUDA (Tested with 12.1, should work on older versions as well)
- C++17
- CMake 3.26
- [https://github.com/nothings/stb/blob/master/stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h)

Installation of these files can be done with the following command on Arch Linux:

```
sudo pacman -S --needed base-devel cuda cmake
```

## How to build

### GPU
Simply run `build.sh` if you are on Linux, the resulting executable will be located in the `build` directory, named `gpgpu_raytrace_rainbow`

### CPU
Switch over to the `cpu-cpp` branch and copy over the `CMakeLists.txt` file or change the following lines in the `main` branch's `CMakeLists.txt`:
```diff
- 2 project(gpgpu_raytrace_rainbow CUDA)
+ 2 project(gpgpu_raytrace_rainbow)
...
- 8 add_executable(gpgpu_raytrace_rainbow rainbow.cu stb_image_write.h)
+ 8 add_executable(gpgpu_raytrace_rainbow rainbow.cpp)
...
- 12 set_target_properties(gpgpu_raytrace_rainbow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```


## Benchmarks & Optimizations

The following table shows the time it takes on different hardware to calculate the refraction & reflection of 300 light vectors (380 nm to 680 nm)

| Hardware              	| Execution time 	| Notes                                        	|
|-----------------------	|----------------	|----------------------------------------------	|
| CPU (Single Threaded) 	| 835 μs         	| g++, no debug symbols, simple for loop       	|
| GPU (Non-Vectorized)  	| 26340 μs       	| nvcc, no debug symbols, simple for loop      	|
| GPU (Vectorized)      	| 86 μs          	| nvcc, no debug symbols, parallel computation 	|

As we can see, the non-vectorized GPU version is much slower than a simple CPU implementation, this is due to the fact that every for loop also calls a `cudaMemcpy` instruction, further slowing down the process.

Calculating the block size and grid size for the 300 vectors, therefore passing everything all data at once reduces the run time by about 99.7% (or in other words, 1/300th of the original runtime, about the same as the amount of data).

A further optimization is removing unnecessary calls to the `normalize` function, as per Khronos's recommendation, `reflect` (and `refract`) should use a normalized vector, if we normalize the vector in these functions, we introduce a huge overhead.
