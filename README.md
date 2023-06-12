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
+ 8 add_executable(gpgpu_raytrace_rainbow rainbow.cpp stb_image_write.h)
...
- 12 set_target_properties(gpgpu_raytrace_rainbow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

## Image display

Running the GPU implementation results in a picture called `rainbow.png` to be placed alongside the source files.
This image is a slice of the simulated world at `z = -3`, with x having a range of `-1.90` to `-2.00` and y having a range of `1.9` to `2.1`.
The explanation of these specific values are, that given the sphere (`(x - 2)^2 + (y + 2)^2 + (z - 1)^2 = 9`, so a center of `(2, -2, 1)` and a radius of `3`) and the initial light (`(3, 2, -3) + t * (0, -1, 1)`, so an initial position of `(3, 2, -3)` with a direction vector of `(0, -1, 1)`) after refraction, reflection & refraction results in these specific values that intersect the `z = -3` plane at a range from `-1.94345` (Ultraviolet light) to `-1.9854` (Red light).

For a more visual representation about the text, check out the following GeoGebra link: [https://www.geogebra.org/m/awdhpswq](https://www.geogebra.org/m/awdhpswq)

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

## Tested On

The following generous people have aided me in testing that the program runs on a variety of different hardware configurations: 

| Person         	| CPU               	| GPU                 	            | OS                         	|
|----------------	|-------------------	|---------------------  	        |----------------------------	|
| Zoltán Balázs  	| Intel Core i7-8700K   | Nvidia GeForce RTX 2080 8GB 	    | Arch Linux (6.3.5-arch1-1) 	|
| Dóra Gregorics 	| AMD Ryzen 5 2600X 	| Nvidia GeForce RTX 2060 6GB 	    | Windows 10 (22H2)          	|
| Márton Petes   	| AMD Ryzen 5 3600  	| Nvidia GeForce GTX 1060 6GB 	    | NixOS 23.11 (6.3.4)        	|
| Márton Petes   	| Intel Core i5-5200U   | Nvidia GeForce 840M 2GB 	        | NixOS 23.05 (6.0.10-zen2)  	|