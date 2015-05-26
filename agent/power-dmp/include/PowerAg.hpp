#ifndef POWERAG_HPP
#define POWERAG_HPP

#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include <functional>
#include <random>
#include <math.h>
#include "arch/AAgent.hpp"
#include "bib/Utils.hpp"
#include "bib/XMLEngine.hpp"
#include <Algo.hpp>
#include <Kernel.hpp>

class PowerAg : public arch::AAgent<> {
 protected:
  void _display(std::ostream& stdout) const override {
    stdout << reward ;
  }
 public:
  PowerAg(unsigned int nb_motors, unsigned int);
  virtual ~PowerAg() {
    delete algo;
  }
  const std::vector<float>& run(float, const std::vector<float>&, bool, bool);
  void start_episode(const std::vector<float>& sensors) override;
  void end_episode() override;
  void save(const std::string& path) override;
  void load(const std::string& path) override;
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override{
        n_steps_max = pt->get<int>("environment.max_step_per_instance");
        n_episodes = pt->get<int>("simulation.max_episode");
        n_instances = pt->get<int>("environment.instance_per_episode");
        state = Eigen::MatrixXf::Zero(n_steps_max,n_sensors+n_motors);
        best_state = Eigen::MatrixXf::Zero(n_steps_max,n_sensors+n_motors);
  }
  Eigen::VectorXf normalDistribution(int size);
  const float PI = 3.14159265358979f;
  private:
  std::vector<float> actuator;
  std::vector< std::pair<float,int>> s_Return;
  unsigned int iter;
  unsigned int episode;
  std::vector<float> rewards;
  float best_value;
  float reward;
  float best_reward;
  float y_max;
  unsigned int pas;
  unsigned int n_steps_max;
  unsigned int n_kernels;
  unsigned int n_motors;
  unsigned int n_sensors;
  unsigned int n_dims;
  unsigned int n_episodes;
  unsigned int n_instances;
  Eigen::MatrixXf best_state;
  Eigen::MatrixXf state;
  Algo *algo;
};

#endif  // POWERAG_HPP
