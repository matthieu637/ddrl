#ifndef util_random_h
#define util_random_h

#include <cstdlib>

namespace panene
{

    inline int rand_int(int high = RAND_MAX, int low = 0) {
        return low + (int)(double(high - low) * (std::rand() / (RAND_MAX + 1.0)));
    }

};
#endif

