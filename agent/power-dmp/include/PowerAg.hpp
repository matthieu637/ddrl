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
  void _dump(std::ostream& stdout) const override {
    stdout << best_reward_episode ;
  }
 public:
  PowerAg(unsigned int nb_motors, unsigned int);
  virtual ~PowerAg() {
    delete algo;
  }
  const std::vector<double>& run(double, const std::vector<double>&, bool, bool);
  void start_episode(const std::vector<double>& sensors) override;
  void end_episode() override;
  void save(const std::string& path) override;
  void load(const std::string& path) override;
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override{
        config.n_steps_max = pt->get<int>("environment.max_step_per_instance");
        config.n_episodes = pt->get<int>("simulation.max_episode");
        config.n_instances = pt->get<int>("environment.instance_per_episode");
        config.var_init = pt->get<double>("agent.var_init");
        config.n_states_per_kernels = pt->get<int>("agent.n_states_per_kernels");
        //config.n_basis_per_dim = bib::to_array<unsigned int>(properties->get<std::string>("agent.n_basis_per_dim"));
        config.n_basis_per_dim = pt->get<int>("agent.n_basis_per_dim");
        config.width_kernel = pt->get<double>("agent.width_kernel");
        config.d_variance = pt->get<double>("agent.d_variance");
        config.elite = pt->get<int>("agent.elite");
        config.elite_variance = pt->get<int>("agent.elite_variance");
        config.n_motors = n_motors;
        config.n_sensors = n_sensors;

        state = Eigen::MatrixXf::Zero(config.n_steps_max+1,n_sensors+n_motors);
        best_state = Eigen::MatrixXf::Zero(config.n_steps_max+1,n_sensors+n_motors);
        algo = new Algo(&config);
        algo->setPointeurIteration(&iter);
        algo->setPointeurEpisode(&episode);
  }
  Eigen::VectorXf normalDistribution(int size);
  const double PI = 3.14159265358979f;

  private:
  std::vector<double> actuator;
  std::vector< std::pair<double,int>> s_Return;
  unsigned int iter;
  unsigned int episode;
  std::vector<double> rewards;
  double best_value;
  double reward;
  double best_reward;
  double best_reward_episode;
  double y_max;
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
