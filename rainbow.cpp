#include <chrono>
#include <cmath>
#include <iostream>

typedef struct {
  double x;
  double y;
  double z;
} double3;

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
double dot(double3 lhs, double3 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

/* Multiply a 3D vector by a scalar */
double3 operator*(double3 vec, double scalar) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

double3 operator*(double scalar, double3 vec) {
  return {vec.x * scalar, vec.y * scalar, vec.z * scalar};
}

/* Subtract two 3D vectors */
double3 operator-(double3 lhs, double3 rhs) {
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

/* Add two 3D vectors */
double3 operator+(double3 lhs, double3 rhs) {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

/* Normalize a 3D vector */
double3 normalize(double3 v) {
  double magnitude =
      std::sqrt(std::pow(v.x, 2) + std::pow(v.y, 2) + std::pow(v.z, 2));
  return {v.x / magnitude, v.y / magnitude, v.z / magnitude};
}

/* Clamps a float between two values */
double clamp(double val, double lower, double upper) {
  return std::max(lower, std::min(val, upper));
}

/* Converts the wavelength in nm to the refractive index of the material, in
 * this case water-air */
double wavelengthToRefraction(double wavelength) {
  return 1.31477 + 0.0108148 / (std::log10(0.00690246 * wavelength));
}

/* Checks if the given point is in the sphere */
bool inSphere(sphere_t sphere, double3 coord) {
  double epsilon = 0.0001;

  return (std::pow((coord.x - sphere.coord.x), 2) +
          std::pow((coord.y - sphere.coord.y), 2) +
          std::pow((coord.z - sphere.coord.z), 2)) <=
         std::pow(sphere.r, 2) + epsilon;
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
Given a normal vector, an incident vector, and a
wavelength, calculates the refracted vector */
double3 refract(double3 N, double3 I, double wavelength, bool inWater) {
  double eta = wavelengthToRefraction(wavelength);
  if (!inWater) {
    eta = 1.0 / eta;
  }
  double k = 1.0 - eta * eta * (1.0 - dot(N, I) * dot(N, I));

  if (k < 0) {
    return {0, 0, 0};
  }

  return eta * I - N * (eta * dot(N, I) + std::sqrt(k));
}

/* Based on https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
Given an incident vector and a normal vector, calculates the reflected vector,
the normal vector must actually be normalzied for optimal results */
double3 reflect(double3 I, double3 N) { return I - 2 * dot(N, I) * N; }

/* Calculates the intersections between a sphere and a radius, if there is
 * any*/
intersection_t vectorSphereIntersection(sphere_t s, light_t l) {
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
  double a = std::pow(l.dir.x, 2) + std::pow(l.dir.y, 2) + std::pow(l.dir.z, 2);

  /* b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
  l.dir.y * (l.coord.y - s.coord.y) + l.dir.z * (l.coord.z - s.coord.z)) */
  double b = 2 * (l.dir.x * (l.coord.x - s.coord.x) +
                  l.dir.y * (l.coord.y - s.coord.y) +
                  l.dir.z * (l.coord.z - s.coord.z));

  // c = (l.coord.x - c.x)^2 + (l.coord.y - c.y)^2 + (l.coord.z - c.z)^2 - r^2
  double c = std::pow((l.coord.x - s.coord.x), 2) +
             std::pow((l.coord.y - s.coord.y), 2) +
             std::pow((l.coord.z - s.coord.z), 2) - std::pow(s.r, 2);

  // discriminant = b^2 - 4 * a * c
  double d = std::pow(b, 2) - 4 * a * c;

  // If the discriminant is negative, there is no solution
  intersection_t i;
  if (d < 0) {
    i.intersects = false;
    return i;
  }

  double t1 = (-1 * b + std::sqrt(d)) / (2 * a);
  double t2 = (-1 * b - std::sqrt(d)) / (2 * a);

  double epsilon = 0.0001;
  double t = 0;

  // If t1 is positive, is smaller than t2 or t2 is negative, we use t1
  // If t2 is positive, is smaller than t1 or t1 is negative, we use t2
  // If both are negative, there is no intersection
  if (0 < t1 && (t1 < t2 || std::abs(t2) <= epsilon)) {
    i.intersects = true;
    t = t1;
  } else if (0 < t2 && (t2 < t1 || std::abs(t1) <= epsilon)) {
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
double3 calculateNormalVector(sphere_t s, double3 i) {
  /* Given a sphere and a point on the sphere's surface, calculate the
   * vector from the sphere's center to the intersection point */
  double3 vector = {i.x - s.coord.x, i.y - s.coord.y, i.z - s.coord.z};

  /* Normalize the given vector */
  return normalize(vector);
}

/* Calculates the angle between two 3D vectors */
double angleBetweenVectors(double3 lhs, double3 rhs) {
  /* Calculate the dot product of the vectors */
  double dotProduct = dot(lhs, rhs);

  /* Calculate the magnitudes of the vectors */
  double magnL =
      std::sqrt(std::pow(lhs.x, 2) + std::pow(lhs.y, 2) + std::pow(lhs.z, 2));
  double magnR =
      std::sqrt(std::pow(rhs.x, 2) + std::pow(rhs.y, 2) + std::pow(rhs.z, 2));

  /* Calculate the angle's cosine between the vectors */
  double cosA = dotProduct / (magnL * magnR);

  /* Return the radians in degrees of the angle between the vectors */
  return acos(cosA);
}

light_t rainbowAirWater(double wavelength) {
  sphere_t sphere = {{2, -2, 1}, 3};
  light_t light = {3, 2, -3, 0, -1, 1, wavelength};

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

  return light;
}

int *wavelengthToRGB(double wavelength) {
  double gamma = 0.80;
  double intensityMax = 255;

  double factor;
  double3 curr_rgb;

  if ((wavelength >= 380) && (wavelength < 440)) {
    curr_rgb.x = -(wavelength - 440) / (440 - 380);
    curr_rgb.y = 0.0;
    curr_rgb.z = 1.0;
  } else if ((wavelength >= 440) && (wavelength < 490)) {
    curr_rgb.x = 0.0;
    curr_rgb.y = (wavelength - 440) / (490 - 440);
    curr_rgb.z = 1.0;
  } else if ((wavelength >= 490) && (wavelength < 510)) {
    curr_rgb.x = 0.0;
    curr_rgb.y = 1.0;
    curr_rgb.z = -(wavelength - 510) / (510 - 490);
  } else if ((wavelength >= 510) && (wavelength < 580)) {
    curr_rgb.x = (wavelength - 510) / (580 - 510);
    curr_rgb.y = 1.0;
    curr_rgb.z = 0.0;
  } else if ((wavelength >= 580) && (wavelength < 645)) {
    curr_rgb.x = 1.0;
    curr_rgb.y = -(wavelength - 645) / (645 - 580);
    curr_rgb.z = 0.0;
  } else if ((wavelength >= 645) && (wavelength < 781)) {
    curr_rgb.x = 1.0;
    curr_rgb.y = 0.0;
    curr_rgb.z = 0.0;
  } else {
    curr_rgb.x = 0.0;
    curr_rgb.y = 0.0;
    curr_rgb.z = 0.0;
  }

  if ((wavelength >= 380) && (wavelength < 420)) {
    factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380);
  } else if ((wavelength >= 420) && (wavelength < 701)) {
    factor = 1.0;
  } else if ((wavelength >= 701) && (wavelength < 781)) {
    factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700);
  } else {
    factor = 0.0;
  }

  return new int[3]{
      curr_rgb.x == 0
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
  const int WAVELENGTHS = 680 - 380;
  double wavelength[WAVELENGTHS];
  for (int i = 0; i < WAVELENGTHS; ++i) {
    wavelength[i] = 380 + i;
  }

  light_t results[WAVELENGTHS];

  auto tS = std::chrono::high_resolution_clock::now();

  for (int i = 380; i < 680; ++i) {
    results[i - 380] = rainbowAirWater(wavelength[i - 380]);
  }

  auto diff = std::chrono::high_resolution_clock::now() - tS;
  std::cout << (ulong)std::chrono::duration_cast<std::chrono::microseconds>(
                   diff)
                   .count()
            << std::endl;

  // for (int i = 0; i < WAVELENGTHS; ++i) {
  //   std::cout << results[i].wavelength << "nm (" << results[i].coord.x << ","
  //             << results[i].coord.y << ", " << results[i].coord.z << ") "
  //             << " -> "
  //             << "(" << results[i].dir.x << ", " << results[i].dir.y << ", "
  //             << results[i].dir.z << ")" << std::endl;
  // }

  return EXIT_SUCCESS;
}
