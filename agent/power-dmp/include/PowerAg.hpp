#ifndef POWERAG_HPP
#define POWERAG_HPP

#include <EigenConfig.h>
#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include <functional>
#include <random>
#include "arch/AAgent.hpp"
#include "bib/Utils.hpp"
#include "bib/XMLEngine.hpp"

#include <math.h>
#include <Kernel.hpp>

class PowerAg : public arch::AAgent<> {
 public:
  PowerAg(unsigned int nb_motors, unsigned int);
  const std::vector<float>& run(float, const std::vector<float>&, bool, bool);
  void start_episode(const std::vector<float>& sensors) override;
  void end_episode() override;
  void save(const std::string& path) override;
  void load(const std::string& path) override;
  Eigen::VectorXf normalDistribution(int size);
  const float PI = 3.14159265358979f;
  private:
  std::vector<float> actuator;
  std::vector< std::pair<double,int>> s_Return;
  int iter;
  std::vector<float> rewards;
  double reward;
  int pas;
  int n_kernels;
  int n_kernels_per_dim;
  int n_dims;
  int n_iter;
  Eigen::MatrixXf param;
  Eigen::VectorXf current_param;
  Eigen::MatrixXf variance;
  Kernel kernel;
};

#endif  // POWERAG_HPP
