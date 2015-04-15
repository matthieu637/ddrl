#ifndef SEED_HPP
#define SEED_HPP

#include <thread>
#include <random>

namespace bib
{

class Seed
{
public:
    static int unifRandInt(int max) {
        std::uniform_int_distribution<int> dis(0, max);
        return dis(engine);
    }
  
    static int unifRandInt(int min, int max) {
        std::uniform_int_distribution<int> dis(min, max);
        return dis(engine);
    }

    static double unifRandFloat(double min, double max) {
        std::uniform_real_distribution<double> dis(min, max);
        return dis(engine);
    }

    static std::mt19937* random_engine() {
        return &engine;
    }
private:
    thread_local static std::mt19937 engine;
};

}  // namespace bib

#endif
