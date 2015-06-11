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
#include <Config.hpp>
#include "bib/IniParser.hpp"

class PowerAg : public arch::AAgent<> {
 protected:
  void _display(std::ostream& stdout) const override {
    stdout << best_reward_episode ;
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
        config.n_steps_max = pt->get<int>("environment.max_step_per_instance");
        config.n_episodes = pt->get<int>("simulation.max_episode");
        config.n_instances = pt->get<int>("environment.instance_per_episode");
        config.var_init = pt->get<float>("agent.var_init");
        config.n_states_per_kernels = pt->get<int>("agent.n_states_per_kernels");


        //config.n_kernels_per_dim = bib::to_array<unsigned int>(properties->get<std::string>("agent.n_kernels_per_dim"));


        config.n_kernels_per_dim = pt->get<int>("agent.n_kernels_per_dim");
        config.width_kernel = pt->get<float>("agent.width_kernel");
        config.d_variance = pt->get<float>("agent.d_variance");
        config.elite = pt->get<int>("agent.elite");
        config.elite_variance = pt->get<int>("agent.elite_variance");
        config.n_motors = n_motors;
        config.n_sensors = n_sensors;
        state = Eigen::MatrixXf::Zero(config.n_steps_max,n_sensors+n_motors);
        best_state = Eigen::MatrixXf::Zero(config.n_steps_max,n_sensors+n_motors);
        algo = new Algo(&config);
        algo->setPointeurIteration(&iter);
        algo->setPointeurEpisode(&episode);
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
  float best_reward_episode;
  float y_max;
  Config config;
  unsigned int pas;
  unsigned int n_motors;
  unsigned int n_sensors;
  unsigned int n_dims;
  Eigen::MatrixXf best_state;
  Eigen::MatrixXf state;
  Algo *algo;
};

#endif  // POWERAG_HPP
