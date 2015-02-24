#include "bib/Utils.hpp"
#include <sys/time.h>
// #include <stdlib.h>
#include <random>
#include <algorithm>

namespace bib {

    float Utils::rand01() {
//     LOG_DEBUG("rand");
        return (float)rand() / (float)RAND_MAX;
    }

    bool Utils::rand01(float limit) {
        if (limit > 0.L) {
            return Utils::rand01() < limit;
        }
        return false;
    }

    float Utils::randin(float a, float b) {
        ASSERT( b > a, "");
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = b - a;
        float r = random * diff;
        return a + r;
    }

    double Utils::abs(double x) {
        if (x > 0)
            return x;
        else return - x;
    }

// a < x < b => c < X < d
    double Utils::transform(double x, double a, double b, double c, double d) {
        if (x < a)
            x = a;
        else if (x > b)
            x = b;

        return c + ((x - a) / (b - a)) * (d - c);
    }

    time_t Utils::srand_mili(bool zero) {
        if (zero) {
            srand(0);
            return 0;
        } else {
            timeval t1;
            gettimeofday(&t1, NULL);
            srand(t1.tv_usec * t1.tv_sec);
            return t1.tv_usec * t1.tv_sec;
        }
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
        return fabs( x1 - x2);
    }


    float Utils::euclidien_dist2D(float x1, float x2, float y1, float y2) {
        return sqrt( pow(x1 - x2, 2) + pow(y1 - y2, 2));
    }

    float Utils::euclidien_dist3D(float x1, float x2, float y1, float y2, float z1, float z2) {
        return sqrt( pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
    }


}
