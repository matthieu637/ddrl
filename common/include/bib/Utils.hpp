#ifndef UTILS_HPP
#define UTILS_HPP

///
///\file Utils.hpp
///\brief les méthodes utiles
///
///

#include "bib/Logger.hpp"
#include <vector>

using std::vector;

namespace bib {

class Utils
{
public :

///
///\brief Retourner 0 ou 1
    static float rand01();

    static bool rand01(float);

    static float randin(float a, float b);
///
///\brief Retourner la valeur absolue de "x"
///\param x : une valeur
    static double abs(double x);

///
///\brief transformer la valeur x qui appartient [a,b] à [c,d]
///\param x :une valeur
///	  a,b : intervalle [a,b]
///	  c,d: intervalle [c,d]
    static double transform(double x, double a, double b, double c, double d);

    static time_t srand_mili(bool zero=false);

    static float* genNrand(int N, float max);


    template<typename T>
    static std::pair<float, float> mean_and_var(const T& _list) {

        float mean = 0.f;
        for(auto it=_list.cbegin(); it != _list.cend(); ++it) {
            float p = *it;
            mean += p;
        }
        mean = (float)(mean/(float)_list.size());

        double variance = 0.f;
        for(auto it=_list.cbegin(); it != _list.cend(); ++it) {
            float p = *it;
            variance += p * p;
        }
        variance = (float)(variance/(float)_list.size());
        variance = variance - mean*mean;

        return {mean, variance};
    }

    static float euclidien_dist1D(float x1, float x2);

    static float euclidien_dist2D(float x1, float x2, float y1, float y2);

    static float euclidien_dist3D(float x1, float x2, float y1, float y2, float z1, float z2);
};

}

#endif // UTILS_HPP
