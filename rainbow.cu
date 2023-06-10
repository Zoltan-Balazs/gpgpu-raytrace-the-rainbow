#include <stdio.h>

const int N = 16;
const int blocksize = 16;

typedef struct {
  float3 coordinates;
  float radius;
} sphere_t;

typedef struct {
  float3 coordinates;
  float3 direction;
  double wavelength;
} light_t;

typedef struct {
  light_t light;
  bool intersects;
} intersection_t;

/* Converts the wavelength in nm to the refractive index of the material, in
 * this case water-air */
__device__ double wavelengthToRefraction(double wavelength) {
  return 1.31477 + 0.0108148 / (log10(0.00690246 * wavelength));
}

/* Checks if the given point is in the sphere */
__device__ bool inSphere(sphere_t sphere, float3 coordinate) {
  double epsilon = 0.0001;

  return abs((sphere.radius * sphere.radius) -
             (pow((coordinate.x - sphere.coordinates.x), 2) +
              pow((coordinate.y - sphere.coordinates.y), 2) +
              pow((coordinate.z - sphere.coordinates.z), 2))) <= epsilon;
}

/* Calculates the dot product of two 3D vectors */
__device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* Multiply a 3D vector by a scalar */
__device__ float3 operator*(float3 vec, float scalar) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}
int main() {
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  float3 *world;
  (void)world;

  char *ad;
  int *bd;
  const int csize = N * sizeof(char);
  const int isize = N * sizeof(int);

  printf("%s", a);

  cudaMalloc((void **)&ad, csize);
  cudaMalloc((void **)&bd, isize);
  cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
  cudaFree(ad);
  cudaFree(bd);

  printf("%s\n", a);
  return EXIT_SUCCESS;
}