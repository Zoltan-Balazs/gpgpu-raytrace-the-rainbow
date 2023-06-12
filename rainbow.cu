#include <cstdlib>
#include <ostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>
#include <cmath>
#include <iostream>

#define CHANNEL_NUM 3
typedef struct {
  double3 coord;
  double r;
} sphere_t;

typedef struct {
  double3 coord;
  double3 dir;
  double wavelength;
} light_t;

typedef struct {
  light_t l;
  bool intersects;
} intersection_t;

/* Calculates the dot product of two 3D vectors */
__device__ double dot(double3 lhs, double3 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/* Multiply a 3D vector by a scalar */
__device__ double3 operator*(double3 vec, double scalar) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

__device__ double3 operator*(double scalar, double3 vec) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

/* Subtract two 3D vectors */
__device__ double3 operator-(double3 lhs, double3 rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

/* Add two 3D vectors */
__device__ double3 operator+(double3 lhs, double3 rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

/* Normalize a 3D vector */
__device__ double3 normalize(double3 v) {
  double magnitude = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
  return {v.x / magnitude, v.y / magnitude, v.z / magnitude};
}

/* Clamps a float between two values */
__device__ double clamp(double val, double lower, double upper) {
  return max(lower, min(val, upper));
}

/* Converts the wavelength in nm to the refractive index of the material, in
 * this case water-air */
__device__ double wavelengthToRefraction(double wavelength) {
  return 1.31477 + 0.0108148 / (log10(0.00690246 * wavelength));
}

/* Checks if the given point is in the sphere */
__device__ bool inSphere(sphere_t sphere, double3 coord) {
  double epsilon = 0.0001;

  return (pow((coord.x - sphere.coord.x), 2) +
          pow((coord.y - sphere.coord.y), 2) +
          pow((coord.z - sphere.coord.z), 2)) <= pow(sphere.r, 2) + epsilon;
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
Given a normal vector, an incident vector, and a
wavelength, calculates the refracted vector */
__device__ double3 refract(double3 N, double3 I, double wavelength,
                           bool inWater) {
  double eta = wavelengthToRefraction(wavelength);
  if (!inWater) {
    eta = 1.0 / eta;
  }
  double k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));

  if (k < 0) {
    return {0, 0, 0};
  }

  return eta * I - (eta * dot(N, I) + sqrt(k)) * N;
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
Given an incident vector and a normal vector, calculates the reflected vector,
the normal vector must actually be normalzied for optimal results */
__device__ double3 reflect(double3 I, double3 N) {
  return I - 2 * dot(N, I) * N;
}

/* Calculates the intersections between a sphere and a radius, if there is
 * any*/
__device__ intersection_t vectorSphereIntersection(sphere_t s, light_t l) {
  /* Given the sphere's center coordinates and radius, and the radius's
   coordinates and direction, we calculate the intersection point:
   (x - s.x)^2 + (y - s.y)^2 + (z - s.z)^2 = s.r^2
   Where `x` is l.coord.x + t * l.dir.x, `y` is l.coord.y + t * l.dir.y,
   `z` is l.coord.z + t * l.dir.z (parametric equation, t is the parameter)
   We then solve for t, and use
   the discriminant to determine if there is an intersection or not.

   The fully expanded eqaution is:
   (l.dir.x^2 + l.dir.y^2 + l.dir.z^2) * t^2 +
   2 * (l.dir.x * (l.coord.x - s.coord.x) + l.dir.y * (l.coord.y - s.coord.y) +
   l.dir.z * (l.coord.z - s.coord.z)) * t +
   (l.coord.x - c.x)^2 + (l.coord.y - c.y)^2 + (l.coord.z - c.z)^2 - r^2 = 0 */

  // a = l.dir.x^2 + l.dir.y^2 + l.dir.z^2
  double a = pow(l.dir.x, 2) + pow(l.dir.y, 2) + pow(l.dir.z, 2);

  /* b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
  l.dir.y * (l.coord.y - s.coord.y) + l.dir.z * (l.coord.z - s.coord.z)) */
  double b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
                  l.dir.y * (l.coord.y - s.coord.y) +
                  l.dir.z * (l.coord.z - s.coord.z));

  // c = (l.coord.x - c.x)^2 + (l.coord.y - c.y)^2 + (l.coord.z - c.z)^2 - r^2
  double c = pow((l.coord.x - s.coord.x), 2) + pow((l.coord.y - s.coord.y), 2) +
             pow((l.coord.z - s.coord.z), 2) - pow(s.r, 2);

  // discriminant = b^2 - 4 * a * c
  double d = pow(b, 2) - 4 * a * c;

  // If the discriminant is negative, there is no solution
  intersection_t i;
  if (d < 0) {
    i.intersects = false;
    return i;
  }

  double t1 = (-1 * b + sqrt(d)) / (2 * a);
  double t2 = (-1 * b - sqrt(d)) / (2 * a);

  double epsilon = 0.0001;
  double t = 0;

  // If t1 is positive, is smaller than t2 or t2 is negative, we use t1
  // If t2 is positive, is smaller than t1 or t1 is negative, we use t2
  // If both are negative, there is no intersection
  if (0 < t1 && (t1 < t2 || abs(t2) <= epsilon)) {
    i.intersects = true;
    t = t1;
  } else if (0 < t2 && (t2 < t1 || abs(t1) <= epsilon)) {
    i.intersects = true;
    t = t2;
  } else {
    i.intersects = false;
  }

  if (i.intersects) {
    i.l = {l.coord.x + t * l.dir.x,
           l.coord.y + t * l.dir.y,
           l.coord.z + t * l.dir.z,
           l.dir.x,
           l.dir.y,
           l.dir.z};
  }

  return i;
}

/* Calculates the normal vector for a sphere and intersection point */
__device__ double3 calculateNormalVector(sphere_t s, double3 i) {
  /* Given a sphere and a point on the sphere's surface, calculate the
   * vector from the sphere's center to the intersection point */
  double3 vector = {i.x - s.coord.x, i.y - s.coord.y, i.z - s.coord.z};

  /* Normalize the given vector */
  return normalize(vector);
}

/* Calculates the angle between two 3D vectors */
__device__ double angleBetweenVectors(double3 lhs, double3 rhs) {
  /* Calculate the dot product of the vectors */
  double dotProduct = dot(lhs, rhs);

  /* Calculate the magnitudes of the vectors */
  double magnL = sqrt(pow(lhs.x, 2) + pow(lhs.y, 2) + pow(lhs.z, 2));
  double magnR = sqrt(pow(rhs.x, 2) + pow(rhs.y, 2) + pow(rhs.z, 2));

  /* Calculate the angle's cosine between the vectors */
  double cosA = dotProduct / (magnL * magnR);

  /* Return the radians in degrees of the angle between the vectors */
  return acos(cosA);
}

__device__ int getGlobalIdx_2D_2D() {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}

__global__ void rainbowAirWater(double *wavelength, light_t *returnVal) {
  unsigned int idx = getGlobalIdx_2D_2D();

  sphere_t sphere = {{2, -2, 1}, 3};
  light_t light = {{3, 2, -3}, {0, -1, 1}, wavelength[idx]};

  intersection_t intersection = vectorSphereIntersection(sphere, light);

  bool refraction = true;
  bool inWater = false;

  for (int i = 0; i < 4 && inSphere(sphere, intersection.l.coord) &&
                  intersection.intersects;
       ++i) {
    double3 normalVector = calculateNormalVector(sphere, intersection.l.coord);
    float angle = clamp(
        angleBetweenVectors(light.dir, intersection.l.coord + normalVector), 0,
        M_PI / 2);

    double3 newVector;
    if (refraction) {
      if (inWater) {
        normalVector = -1 * normalVector;
      }

      newVector =
          refract(normalVector, intersection.l.dir, light.wavelength, inWater);
      inWater = true;
      refraction = false;
    } else {
      newVector = reflect(intersection.l.dir, -1 * normalVector);
      refraction = true;
    }
    light = {intersection.l.coord, newVector, light.wavelength};
    intersection = vectorSphereIntersection(sphere, light);
  }

  returnVal[idx] = light;
}

__global__ void wavelengthToRGB(double *wavelength, int3 *rgb) {
  unsigned int idx = getGlobalIdx_2D_2D();

  double gamma = 0.80;
  double intensityMax = 255;

  double factor;
  float3 curr_rgb;

  if ((wavelength[idx] >= 380) && (wavelength[idx] < 440)) {
    curr_rgb.x = -(wavelength[idx] - 440) / (440 - 380);
    curr_rgb.y = 0.0;
    curr_rgb.z = 1.0;
  } else if ((wavelength[idx] >= 440) && (wavelength[idx] < 490)) {
    curr_rgb.x = 0.0;
    curr_rgb.y = (wavelength[idx] - 440) / (490 - 440);
    curr_rgb.z = 1.0;
  } else if ((wavelength[idx] >= 490) && (wavelength[idx] < 510)) {
    curr_rgb.x = 0.0;
    curr_rgb.y = 1.0;
    curr_rgb.z = -(wavelength[idx] - 510) / (510 - 490);
  } else if ((wavelength[idx] >= 510) && (wavelength[idx] < 580)) {
    curr_rgb.x = (wavelength[idx] - 510) / (580 - 510);
    curr_rgb.y = 1.0;
    curr_rgb.z = 0.0;
  } else if ((wavelength[idx] >= 580) && (wavelength[idx] < 645)) {
    curr_rgb.x = 1.0;
    curr_rgb.y = -(wavelength[idx] - 645) / (645 - 580);
    curr_rgb.z = 0.0;
  } else if ((wavelength[idx] >= 645) && (wavelength[idx] < 781)) {
    curr_rgb.x = 1.0;
    curr_rgb.y = 0.0;
    curr_rgb.z = 0.0;
  } else {
    curr_rgb.x = 0.0;
    curr_rgb.y = 0.0;
    curr_rgb.z = 0.0;
  }

  if ((wavelength[idx] >= 380) && (wavelength[idx] < 420)) {
    factor = 0.3 + 0.7 * (wavelength[idx] - 380) / (420 - 380);
  } else if ((wavelength[idx] >= 420) && (wavelength[idx] < 701)) {
    factor = 1.0;
  } else if ((wavelength[idx] >= 701) && (wavelength[idx] < 781)) {
    factor = 0.3 + 0.7 * (780 - wavelength[idx]) / (780 - 700);
  } else {
    factor = 0.0;
  }

  rgb[idx] = {curr_rgb.x == 0
                  ? 0
                  : (int)round(intensityMax * pow(curr_rgb.x * factor, gamma)),
              curr_rgb.y == 0
                  ? 0
                  : (int)round(intensityMax * pow(curr_rgb.y * factor, gamma)),
              curr_rgb.z == 0
                  ? 0
                  : (int)round(intensityMax * pow(curr_rgb.z * factor, gamma))};
}

int main() {
  // Wavelengths from 380nm to 680nm on the CPU
  const int WAVELENGTHS = 680 - 380;
  double wavelength[WAVELENGTHS];
  for (int i = 0; i < WAVELENGTHS; ++i) {
    wavelength[i] = 380 + i;
  }

  // Variables for use on the GPU & CPU
  double *gpu_wavelength;
  light_t *cpu_results, *gpu_results;
  int3 *cpu_rgb, *gpu_rgb;

  // Grid & Block size for the kernel, 3 * 4 * 5 * 5 = 300 threads, the 300
  // wavelengths
  dim3 block_size(3, 4);
  dim3 grid_size(5, 5);

  // Allocate memory on the GPU & CPU
  cudaError_t cudaError =
      cudaMalloc((void **)&gpu_results, WAVELENGTHS * sizeof(light_t));
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating memory on GPU: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  cudaError =
      cudaHostAlloc((void **)&cpu_results, WAVELENGTHS * sizeof(light_t),
                    cudaHostAllocDefault);
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating pinned memory: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  cudaError =
      cudaMalloc((void **)&gpu_wavelength, WAVELENGTHS * sizeof(double));
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating pinned memory: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }
  cudaMemcpy(gpu_wavelength, wavelength, WAVELENGTHS * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaError = cudaMalloc((void **)&gpu_rgb, WAVELENGTHS * sizeof(int3));
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating pinned memory: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  cudaError = cudaHostAlloc((void **)&cpu_rgb, WAVELENGTHS * sizeof(int3),
                            cudaHostAllocDefault);
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating pinned memory: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  // Clock for timing
  // auto tS = std::chrono::high_resolution_clock::now();

  // Run the kernel for the rainbow vector calculation
  rainbowAirWater<<<block_size, grid_size>>>(gpu_wavelength, gpu_results);

  // Copy back the result to the CPU
  cudaMemcpy(cpu_results, gpu_results, WAVELENGTHS * sizeof(light_t),
             cudaMemcpyDeviceToHost);

  // Print the timing
  // auto diff = std::chrono::high_resolution_clock::now() - tS;
  // std::cout << (ulong)std::chrono::duration_cast<std::chrono::microseconds>(
  //                  diff)
  //                  .count()
  //           << std::endl;

  // Print the refraction, reflection results
  // for (int i = 0; i < WAVELENGTHS; ++i) {
  //   std::cout << cpu_results[i].wavelength << "nm (" <<
  //   cpu_results[i].coord.x
  //             << ", " << cpu_results[i].coord.y << ", "
  //             << cpu_results[i].coord.z << ") "
  //             << " -> "
  //             << "(" << cpu_results[i].dir.x << ", " << cpu_results[i].dir.y
  //             << ", " << cpu_results[i].dir.z << ")" << std::endl;
  // }

  // Run the kernel for the wavelength -> RGB conversion
  wavelengthToRGB<<<block_size, grid_size>>>(gpu_wavelength, gpu_rgb);

  // Copy back the result to the CPU
  cudaMemcpy(cpu_rgb, gpu_rgb, WAVELENGTHS * sizeof(int3),
             cudaMemcpyDeviceToHost);

  /* Calculate the image, since we are using the z = -3.0 plane, our x values
   * will range from -1.94345 (Ultraviolet light) to -1.9854 (Red light)
   * Since this change from -1.94345 to -1.9854 is 300 (the number of
   * wavelengths), we can calculate the resolution for -1.90 to -2.00, which is
   * 715 300 is used for the height, since we take 3 pixel at just before 2.0,
   * at 2.0 and just after 2.0 for the y value
   */
  const int WIDTH = 715;
  const int HEIGHT = 300;

  unsigned char *pixels = new unsigned char[WIDTH * HEIGHT * CHANNEL_NUM];
  memset(pixels, 255, WIDTH * HEIGHT * CHANNEL_NUM * sizeof(unsigned char));

  double zPlane = -3.0;
  // Used for indexing the pixel array
  int idx = 0;
  for (int j = 0; j < HEIGHT; ++j) {
    for (int i = 0; i < WIDTH; ++i) {
      // Current x and y values based on the resolution we specified
      double currentX = -1.90 + (i * -0.000140161);
      double currentY = 1.9 + (int)(j / 100) * 0.1;
      double epsilon = 0.001;
      bool inRange = false;
      /* Iterate over the vector results and find the vector that intersects
       * the plane at the current x and y values (if any) */
      for (int k = 0; k < WAVELENGTHS; ++k) {
        /* t is used for the parametric equation of the line
         * we need to calculate the t such that
         * cpu_results[k].coord.z + t * cpu_results[k].dir.z = -3.0 */
        double t = (zPlane - cpu_results[k].coord.z) / cpu_results[k].dir.z;
        /* If for the given t, the x and y values are within epsilon of the
         * current calculated x and y values, we take the wavelength of the
         * vector and convert it to RGB */
        if (abs(cpu_results[k].coord.x + t * cpu_results[k].dir.x - currentX) <=
                epsilon &&
            abs(cpu_results[k].coord.y + t * cpu_results[k].dir.y - currentY) <=
                epsilon) {
          pixels[idx++] = cpu_rgb[(int)cpu_results[k].wavelength - 380].x;
          pixels[idx++] = cpu_rgb[(int)cpu_results[k].wavelength - 380].y;
          pixels[idx++] = cpu_rgb[(int)cpu_results[k].wavelength - 380].z;
          inRange = true;
          break;
        }
      }
      if (!inRange) {
        // Else we use white
        pixels[idx++] = 255;
        pixels[idx++] = 255;
        pixels[idx++] = 255;
      }
    }
  }

  // Write the image to the file
  stbi_write_png("rainbow.png", WIDTH, HEIGHT, CHANNEL_NUM, pixels,
                 WIDTH * CHANNEL_NUM);

  // Free up memory
  cudaFreeHost(cpu_rgb);
  cudaFreeHost(cpu_results);
  cudaFree(gpu_wavelength);
  cudaFree(gpu_results);
  cudaFree(gpu_rgb);

  return EXIT_SUCCESS;
}