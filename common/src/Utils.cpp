#include "bib/Utils.hpp"

#include <sys/time.h>
#include <random>
#include <algorithm>
#include <functional>

#include "bib/Seed.hpp"


namespace bib {

double Utils::rand01() {
  return bib::Seed::unifRandFloat(0., 1.);
}

bool Utils::rand01(double limit) {
  if (limit > 0.L) {
    return Utils::rand01() < limit;
  }
  return false;
}

double Utils::randin(double a, double b) {
  ASSERT(b > a, "first argument " << a << " sould be smaller than second :" << b);
  double random = rand01();
  double diff = b - a;
  double r = random * diff;
  return a + r;
}

bool Utils::randBool() {
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

double* Utils::genNrand(int N, double max) {
  double* tab = new double[N];
  tab[0] = 0.;

  for (int i = 1; i < N; i++)
    tab[i] = rand01() * max;

  std::sort(tab, tab + N, std::less<double>());
  return tab;
}

double Utils::euclidien_dist1D(double x1, double x2) {
  return fabs(x1 - x2);
}

double Utils::euclidien_dist2D(double x1, double x2, double y1, double y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double Utils::euclidien_dist3D(double x1, double x2, double y1, double y2, double z1,
                              double z2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

double Utils::euclidien_dist(const std::vector<double>& v1, const std::vector<double>& v2, double dmax) {
  double sum = 0;
  for(uint i=0; i < v1.size(); i++)
    sum += pow(v1[i] - v2[i], 2);

  return sqrt(sum) / (v1.size() * dmax);
}

double Utils::euclidien_dist_ref(const std::vector<double>& v1, double refp) {
  double sum = 0;
  for(uint i=0; i < v1.size(); i++)
    sum += pow(v1[i] - refp, 2);

  return sqrt(sum) / (v1.size());
}

}  // namespace bib
