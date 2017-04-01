#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <vector>

#include "arch/AAgent.hpp"
#include "arch/AEnvironment.hpp"
#include "../bib/Utils.hpp"
#include "bib/Utils.hpp"

namespace arch {

class ExampleEnv : public arch::AEnvironment<> {
 public:
  ExampleEnv() : internal_state(6) {
    for (unsigned int i = 0; i < internal_state.size(); i++)
      internal_state[i] = bib::Utils::randin(-1, 1);
  }
  const std::vector<double>& perceptions() const {
    return internal_state;
  }
  unsigned int number_of_actuators() const {
    return 3;
  }
  unsigned int number_of_sensors() const {
    return internal_state.size();
  }
  double performance() const {
    return 0;
  }
  void _apply(const std::vector<double>&) {}

  std::vector<double> internal_state;
};

class ExampleAgent : public arch::AAgent<> {
 public:
  ExampleAgent(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<double>& run(double, const std::vector<double>&, bool, bool) override {
    time_for_ac--;
    if (time_for_ac == 0) {
      time_for_ac = decision_each;
      for (unsigned int i = 0; i < actuator.size(); i++)
        actuator[i] = bib::Utils::randin(-1, 1);
    }
    return actuator;
  }
  
  void start_episode(const std::vector<double>&, bool) override {
    time_for_ac = 1;
  }

  virtual ~ExampleAgent() {
  }
  
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    decision_each                       = pt->get<int>("agent.decision_each");
  }

  int decision_each, time_for_ac;
  std::vector<double> actuator;
};

class ZeroAgent : public arch::AAgent<> {
public:
  ZeroAgent(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<double>& run(double, const std::vector<double>&, bool, bool) override {
    return actuator;
  }
  
  virtual ~ZeroAgent() {
  }
  
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    for (unsigned int i = 0; i < actuator.size(); i++)
      actuator[i] = 0;
  }
  std::vector<double> actuator;
};

template <typename T>
T gaussian(T x, T m, T s)
{
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - m) / s;

    return (inv_sqrt_2pi / s) * std::exp(-T(0.5f) * a * a);
}

class SimpleEnv1D : public arch::AEnvironment<> {
 public:
  SimpleEnv1D() : s(1), add_time_in_state(false) {
//       s[0] = -0.7;
//    s[0] = bib::Utils::randin(-1, 1);
  }
  
  void _next_instance() override {_reset_episode();}
  
  void _next_instance_choose(const std::vector<double>& sto) override {
    _reset_episode_choose(sto);
  }
  
  void _reset_episode() override {
    s[0] = bib::Utils::randin(-1, 1);
    while(final_state())
	s[0] = bib::Utils::randin(-1, 1);
    
    if(add_time_in_state)
      s[1] = -1.f;

    first_state_stochasticity.resize(1);
    first_state_stochasticity[0] = s[0];
  }
  
  void _reset_episode_choose(const std::vector<double>& stochasticity) override {
    s[0] = stochasticity[0];
    if(add_time_in_state)
      s[1] = -1.f;
  }
  
  const std::vector<double>& perceptions() const {
    return s;
  }
  
  unsigned int number_of_actuators() const {
    return 1;
  }
  
  bool final_state() const override {
    return performance() >= 0.;
  }
  
  unsigned int number_of_sensors() const {
    return s.size();
  }
  
  double performance() const {
//     y_1\left(x,b,a\right)=\frac{1}{a\sqrt{2\pi }}e^{-\frac{\left(x-b\right)^2}{2a^2}}
//     -4+y_1\left(x,\ 0.7,\ 0.05\right)+y_1\left(x,\ 1,\ 0.1\right)+10\cdot y_1\left(x,\ -0.7,\ 1\right)
//     return -4 + gaussian<double>(s[0], 0.7f, 0.05) + gaussian<double>(s[0], 1.f, 0.1f) + 10.f * gaussian<double>(s[0], -0.7f, 1.f);
    return -1.f + gaussian<double>(s[0], 0.7f, 0.05);
  }
  
  void _apply(const std::vector<double>& a) {
    s[0] = s[0] + a[0]/ 2.f; 
    if( s[0] > 1.f)
      s[0] = 1.f;
    else if( s[0] < -1.f)
      s[0] = -1.f;
    
    if(add_time_in_state)
      s[1] = bib::Utils::transform(current_step, 0, max_step_per_instance, -1.f, 1.f);
  }
  
  void unique_invoke(boost::property_tree::ptree* inifile, boost::program_options::variables_map*) {
    instance_per_episode    = 1;
    max_step_per_instance   = 50;
    
    if(inifile->count("environment.add_time_in_state") != 0 )
      if(inifile->get<bool>("environment.add_time_in_state")){
          s.resize(2);
          add_time_in_state = true;
      }
  }

  std::vector<double> s;
  bool add_time_in_state;
};

// -1 -0.5(-1) 1(-1) 0(-1)
// -1 -0.5(-1) 1(-1) 0(-1)
// -1 0.25(-1) 0.5(1)
class SimpleEnv1DFixedTraj : public arch::AEnvironment<> {
 public:
  SimpleEnv1DFixedTraj() : s({{-1}, {-0.5}, {1}, {0}}), s_goal({{-1}, {0.25}, {0.5}}) {
    current_step = 0;
    current_episode = -1;
  }
  
  void _reset_episode() {
    current_step = 0;
    current_episode++;
  }
  
  const std::vector<double>& perceptions() const {   
    if(current_episode <= 1)
      return s[current_step];
    
    return s_goal[current_step];
  }
  
  unsigned int number_of_actuators() const {
    return 1;
  }
  
  bool final_state() const override {
    if(current_episode <= 1)
      return false;
    
    return current_step >= 2;
  }
  
  unsigned int number_of_sensors() const {
    return 1;
  }
  
  double performance() const {
    if(current_episode <= 1 || current_step < 2)
      return -1.f;
    
    return 1.f;
  }
  
  void _apply(const std::vector<double>&) {
    current_step++;
  }
  
  void unique_invoke(boost::property_tree::ptree*, boost::program_options::variables_map*) {
    instance_per_episode    = 1;
    max_step_per_instance   = 3;
  }

  const std::vector< std::vector<double> > s;
  const std::vector< std::vector<double> > s_goal;
  uint current_step;
  uint current_episode;
};

}  // namespace arch

#endif  // EXAMPLE_H
