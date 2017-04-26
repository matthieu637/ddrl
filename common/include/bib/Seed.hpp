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
    return dis(seed_instance->engine);
  }

  /**
   * [min; max]
   **/
  static int unifRandInt(int min, int max) {
    std::uniform_int_distribution<int> dis(min, max);
    return dis(seed_instance->engine);
  }

  static double unifRandFloat(double min, double max) {
    std::uniform_real_distribution<double> dis(min, max);
    return dis(seed_instance->engine);
  }
  
  template<typename Real>
  static Real gaussianRand(Real mean, Real sigma){
    std::normal_distribution<Real> dis(mean, sigma);
    return dis(seed_instance->engine);
  }
  
  static void setFixedSeedUTest(){
    LOG_INFO("WARNING: FIXED SEED SET (thread " << std::this_thread::get_id() << " )");
    seed_instance->engine.seed(0);
    caffe::Caffe::set_random_seed(0);
  }

  static std::mt19937* random_engine() {
    return &(seed_instance->engine);
  }
  
private:
  Seed() : engine(std::clock() + std::hash<std::thread::id>()(std::this_thread::get_id())){
    caffe::Caffe::set_random_seed(engine());
  }
private:
  std::mt19937 engine;
  thread_local static std::shared_ptr<Seed> seed_instance; //several instance for each thread
};

}  // namespace bib

#endif
