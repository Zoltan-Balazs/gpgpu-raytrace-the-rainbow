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

Simply run `build.sh` if you are on Linux, the resulting executable will be located in the `build` directory, named `gpgpu_raytrace_rainbow`