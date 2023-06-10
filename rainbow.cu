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