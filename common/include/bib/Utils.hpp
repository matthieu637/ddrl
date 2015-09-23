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
  static float rand01();

  static bool rand01(float);

  static float randin(float a, float b);

  static bool randBool();

  ///
  // /\brief transformer la valeur x qui appartient [a,b] à [c,d]
  // /\param x :une valeur
  ///   a,b : intervalle [a,b]
  ///   c,d: intervalle [c,d]
  static double transform(double x, double a, double b, double c, double d);

  static float *genNrand(int N, float max);

  typedef struct {
    float var;
    float mean;
    float max;
    float min;
  } V3M;

  template <typename T>
  static V3M statistics(const T &_list) {
    float mean = 0.f;
    float min = *_list.cbegin();
    float max = *_list.cbegin();
    for (auto it = _list.cbegin(); it != _list.cend(); ++it) {
      float p = *it;
      mean += p;

      if (p > max)
        max = p;
      else if (p < min)
        min = p;
    }
    mean = (mean / static_cast<float>(_list.size()));

    float variance = 0.f;
    for (auto it = _list.cbegin(); it != _list.cend(); ++it) {
      float p = *it;
      variance += p * p;
    }
    variance = (variance / static_cast<float>(_list.size()));
    variance = variance - mean * mean;

    return {variance, mean, max, min};
  }

  static float euclidien_dist1D(float x1, float x2);

  static float euclidien_dist2D(float x1, float x2, float y1, float y2);

  static float euclidien_dist3D(float x1, float x2, float y1, float y2,
                                float z1, float z2);
  
  static float euclidien_dist(const std::vector<float>& v1, const std::vector<float>& v2, float dmax);
  
  static float euclidien_dist_ref(const std::vector<float>& v1, float refp);
};
}  // namespace bib

#endif  // UTILS_HPP
