#include "bib/Utils.hpp"

#include <sys/time.h>
#include <random>
#include <algorithm>
#include <functional>

#include "bib/Seed.hpp"


namespace bib {

float Utils::rand01() {
  return static_cast<float>(RAND()) / static_cast<float>(RAND_MAX);
}
bool Utils::rand01(float limit) {
  if (limit > 0.L) {
    return Utils::rand01() < limit;
  }
  return false;
}

float Utils::randin(float a, float b) {
  ASSERT(b > a, "first argument " << a << " sould be smaller than second :" << b);
  float random = rand01();
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

bool Utils::randBool(){
  return Utils::rand01() < 0.5;
}

// a < x < b => c < X < d
double Utils::transform(double x, double a, double b, double c, double d) {
  if (x < a)
    x = a;
  else if (x > b)
    x = b;

  return c + ((x - a) / (b - a)) * (d - c);
}

float* Utils::genNrand(int N, float max) {
  float* tab = new float[N];
  tab[0] = 0.;

  for (int i = 1; i < N; i++)
    tab[i] = rand01() * max;

  std::sort(tab, tab + N, std::less<float>());
  return tab;
}

float Utils::euclidien_dist1D(float x1, float x2) {
  return fabs(x1 - x2);
}

float Utils::euclidien_dist2D(float x1, float x2, float y1, float y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

float Utils::euclidien_dist3D(float x1, float x2, float y1, float y2, float z1,
                              float z2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}
}  // namespace bib
