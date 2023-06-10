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

/* Calculates the dot product of two 3D vectors */
__device__ float dot(float3 lhs, float3 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/* Multiply a 3D vector by a scalar */
__device__ float3 operator*(float3 vec, float scalar) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

__device__ float3 operator*(float scalar, float3 vec) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

/* Subtract two 3D vectors */
__device__ float3 operator-(float3 lhs, float3 rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

/* Normalize a 3D vector */
__device__ float3 normalize(float3 v) {
  float magnitude = sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
  return {v.x / magnitude, v.y / magnitude, v.z / magnitude};
}

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

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
Given a normal vector, an incident vector, and a
wavelength, calculates the refracted vector */
__device__ float3 refract(float3 N, float3 I, double wavelength) {
  float eta = wavelengthToRefraction(wavelength);
  eta = 1.0 / eta;
  float k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));

  if (k < 0) {
    return {0, 0, 0};
  }

  return eta * I - N * (eta * dot(N, I) + sqrt(k));
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
Given an incident vector and a normal vector, calculates the reflected vector,
the normal vector must actually be normalzied for optimal results */
__device__ float3 reflect(float3 I, float3 N) {
  return I - 2 * dot(normalize(N), I) * normalize(N);
}
}

/* Calculates the normal vector for a sphere and intersection point */
__device__ float3 calculateNormalVector(sphere_t sphere,
                                        float3 intersectionPoint) {
  /* Given a sphere and a point on the sphere's surface, calculate the
   * vector from the sphere's center to the intersection point */
  float3 vector = {intersectionPoint.x - sphere.coordinates.x,
                   intersectionPoint.y - sphere.coordinates.y,
                   intersectionPoint.z - sphere.coordinates.z};

  /* Normalize the given vector */
  return normalize(vector);
}

/* Calculates the angle between two 3D vectors */
__device__ float angleBetweenVectors(float3 lhs, float3 rhs) {
  /* Calculate the dot product of the vectors */
  float dotProduct = dot(lhs, rhs);

  /* Calculate the magnitudes of the vectors */
  float magnL = sqrt(pow(lhs.x, 2) + pow(lhs.y, 2) + pow(lhs.z, 2));
  float magnR = sqrt(pow(rhs.x, 2) + pow(rhs.y, 2) + pow(rhs.z, 2));

  /* Calculate the angle's cosine between the vectors */
  float cosA = dotProduct / (magnL * magnR);

  /* Return the radians in degrees of the angle between the vectors */
  return acos(cosA);
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
Given a normal vector, an incident vector, and a
wavelength, calculates the refracted vector */
__device__ float3 refract(float3 N, float3 I, double wavelength) {
  float eta = wavelengthToRefraction(wavelength);
  eta = 1.0 / eta;
  float k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));

  if (k < 0) {
    return {0, 0, 0};
  }

  return eta * I - N * (eta * dot(N, I) + sqrt(k));
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
Given an incident vector and a normal vector, calculates the reflected vector,
the normal vector must actually be normalzied for optimal results */
__device__ float3 reflect(float3 I, float3 N) { return I - 2 * dot(N, I) * N; }

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