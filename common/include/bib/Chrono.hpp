#ifndef CHRONO_H
#define CHRONO_H

#include <chrono>

using namespace std::chrono;

namespace bib {

class Chrono
{
public:
    void start();
    double finish();
    void reset();
private:

    high_resolution_clock::time_point begin;
};

}

#endif // CHRONO_H
