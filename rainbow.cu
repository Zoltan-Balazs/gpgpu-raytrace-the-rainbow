#include <chrono>
#include <cmath>
#include <iostream>

const int blocksize = 16;

typedef struct {
  float3 coord;
  float r;
} sphere_t;

typedef struct {
  float3 coord;
  float3 dir;
  double wavelength;
} light_t;

typedef struct {
  light_t l;
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
__device__ bool inSphere(sphere_t sphere, float3 coord) {
  double epsilon = 0.0001;

  return abs((sphere.r * sphere.r) - (pow((coord.x - sphere.coord.x), 2) +
                                      pow((coord.y - sphere.coord.y), 2) +
                                      pow((coord.z - sphere.coord.z), 2))) <=
         epsilon;
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
  float a = pow(l.dir.x, 2) + pow(l.dir.y, 2) + pow(l.dir.z, 2);

  /* b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
  l.dir.y * (l.coord.y - s.coord.y) + l.dir.z * (l.coord.z - s.coord.z)) */
  float b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
                 l.dir.y * (l.coord.y - s.coord.y) +
                 l.dir.z * (l.coord.z - s.coord.z));

  // c = (l.coord.x - c.x)^2 + (l.coord.y - c.y)^2 + (l.coord.z - c.z)^2 - r^2
  float c = pow((l.coord.x - s.coord.x), 2) + pow((l.coord.y - s.coord.y), 2) +
            pow((l.coord.z - s.coord.z), 2) - pow(s.r, 2);

  // discriminant = b^2 - 4 * a * c
  float d = pow(b, 2) - 4 * a * c;

  // If the discriminant is negative, there is no solution
  intersection_t i;
  if (d < 0) {
    i.intersects = false;
    return i;
  }

  float t1 = (-1 * b + sqrt(d)) / (2 * a);
  float t2 = (-1 * b - sqrt(d)) / (2 * a);

  float t = 0;

  // If t1 is positive, is smaller than t2 or t2 is negative, we use t1
  // If t2 is positive, is smaller than t1 or t1 is negative, we use t2
  // If both are negative, there is no intersection
  if (0 < t1 && (t1 < t2 || t2 <= 0)) {
    i.intersects = true;
    t = t1;
  } else if (0 < t2 && (t2 < t1 || t1 <= 0)) {
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
__device__ float3 calculateNormalVector(sphere_t s, float3 i) {
  /* Given a sphere and a point on the sphere's surface, calculate the
   * vector from the sphere's center to the intersection point */
  float3 vector = {i.x - s.coord.x, i.y - s.coord.y, i.z - s.coord.z};

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

__global__ void test(sphere_t sphere, light_t light, float *returnVal) {
  intersection_t intersection = vectorSphereIntersection(sphere, light);

  int i = 0;
  bool refraction = true;

  for (i = 0; i < 4 && inSphere(sphere, intersection.l.coord) &&
              intersection.intersects;
       ++i) {
    float3 normalVector = calculateNormalVector(sphere, intersection.l.coord);
    float angle = angleBetweenVectors(light.dir, normalVector);
    if (refraction) {
      light.dir = refract(normalVector, intersection.l.dir, light.wavelength);
      refraction = false;
    } else {
      light.dir = reflect(intersection.l.dir, normalVector);
      refraction = true;
    }
    light.coord = intersection.l.coord;
    intersection = vectorSphereIntersection(sphere, light);
  }

  returnVal[0] = intersection.l.coord.x;
  returnVal[1] = intersection.l.coord.y;
  returnVal[2] = intersection.l.coord.z;
}

int main() {
  sphere_t sphere = {{2, -2, 1}, 3};

  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  float *hostVal = 0;
  float *val;

  cudaError_t cudaError = cudaMalloc((void **)&val, 3 * sizeof(float));
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating memory on GPU: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  cudaError =
      cudaHostAlloc((void **)&hostVal, 3 * sizeof(float), cudaHostAllocDefault);
  if (cudaError != cudaSuccess) {
    std::cout << "Error while allocating pinned memory: "
              << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }
  cudaMemcpy(val, hostVal, 3 * sizeof(float), cudaMemcpyHostToDevice);

  auto tS = std::chrono::high_resolution_clock::now();

  for (int i = 380; i < 740; ++i) {
    test<<<dimGrid, dimBlock>>>(sphere, {3, 2, -3, 0, -1, 1, (double)i}, val);

    cudaMemcpy(hostVal, val, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "(" << hostVal[0] << ", " << hostVal[1] << ", " << hostVal[2]
              << ")\n";
  }

  auto diff = std::chrono::high_resolution_clock::now() - tS;
  std::cout << (ulong)std::chrono::duration_cast<std::chrono::microseconds>(
                   diff)
                   .count()
            << std::endl;

  return EXIT_SUCCESS;
}