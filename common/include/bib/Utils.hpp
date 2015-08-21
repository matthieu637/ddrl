#ifndef UTILS_HPP
#define UTILS_HPP

///
// /\file Utils.hpp
// /\brief les méthodes utiles
///
///

#include <vector>
#include <algorithm>

#include "bib/Logger.hpp"

using std::vector;

namespace bib {

class Utils {
 public:
  ///
  // /\brief Retourner 0 ou 1
  static double rand01();

  static bool rand01(double);

  static double randin(double a, double b);

  static bool randBool();

  ///
  // /\brief transformer la valeur x qui appartient [a,b] à [c,d]
  // /\param x :une valeur
  ///   a,b : intervalle [a,b]
  ///   c,d: intervalle [c,d]
  static double transform(double x, double a, double b, double c, double d);

  static double *genNrand(int N, double max);

  typedef struct {
    double var;
    double mean;
    double max;
    double min;
  } V3M;

  template <typename T>
  static V3M statistics(const T &_list) {
    double mean = 0.f;
    double min = *_list.cbegin();
    double max = *_list.cbegin();
    for (auto it = _list.cbegin(); it != _list.cend(); ++it) {
      double p = *it;
      mean += p;

      if (p > max)
        max = p;
      else if (p < min)
        min = p;
    }
    mean = (mean / static_cast<double>(_list.size()));

    double variance = 0.f;
    for (auto it = _list.cbegin(); it != _list.cend(); ++it) {
      double p = *it;
      variance += p * p;
    }
    variance = (variance / static_cast<double>(_list.size()));
    variance = variance - mean * mean;

    return {variance, mean, max, min};
  }

  static double euclidien_dist1D(double x1, double x2);

  static double euclidien_dist2D(double x1, double x2, double y1, double y2);

  static double euclidien_dist3D(double x1, double x2, double y1, double y2,
                                double z1, double z2);

  static double euclidien_dist(const std::vector<double>& v1, const std::vector<double>& v2, double dmax);

  static double euclidien_dist_ref(const std::vector<double>& v1, double refp);
};
}  // namespace bib

#endif  // UTILS_HPP
