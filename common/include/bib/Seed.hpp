#ifndef SEED_HPP
#define SEED_HPP

#include <thread>
#include <random>
#include "bib/Logger.hpp"
#include "caffe/caffe.hpp"

namespace bib {

class Seed {
public:

  /**
   * [0; max]
   **/
  static int unifRandInt(int max) {
    std::uniform_int_distribution<int> dis(0, max);
    return dis(engine);
  }

  /**
   * [min; max]
   **/
  static int unifRandInt(int min, int max) {
    std::uniform_int_distribution<int> dis(min, max);
    return dis(engine);
  }

  static double unifRandFloat(double min, double max) {
    std::uniform_real_distribution<double> dis(min, max);
    return dis(engine);
  }
  
  template<typename Real>
  static Real gaussianRand(Real mean, Real sigma){
    std::normal_distribution<Real> dis(mean, sigma);
    return dis(engine);
  }
  
  static void setFixedSeedUTest(){
    LOG_INFO("WARNING: FIXED SEED SET");
    engine.seed(0);
    caffe::Caffe::set_random_seed(0);
  }

  static std::mt19937* random_engine() {
    return &engine;
  }
  
private:
  Seed(){
    caffe::Caffe::set_random_seed(engine());
  }
private:
  thread_local static std::mt19937 engine;
  static Seed seed_instance;
};

}  // namespace bib

#endif
