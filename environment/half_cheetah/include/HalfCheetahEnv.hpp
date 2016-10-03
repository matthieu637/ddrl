#ifndef HALTCHEETAHENV_H
#define HALTCHEETAHENV_H

#include <string>
#include <vector>

#include "bib/IniParser.hpp"
#include "arch/AEnvironment.hpp"
#include "HalfCheetahWorld.hpp"
#include "HalfCheetahWorldView.hpp"

class HalfCheetahEnv : public arch::AEnvironment<> {
 public:
  HalfCheetahEnv() {
    ODEFactory::getInstance();
    instance = nullptr;
  }

  ~HalfCheetahEnv() {
    delete instance;
  }

  const std::vector<double>& perceptions() const {
    return instance->state();
  }

  double performance() const {
    return reward;
  }

  bool final_state() const {
    return instance->final_state();
  }

  unsigned int number_of_actuators() const {
    return instance->activated_motors();
  }
  unsigned int number_of_sensors() const {
    return instance->state().size();
  }

 private:
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* vm) {
    hcheetah_physics init;
    init.apply_armature = pt->get<bool>("environment.apply_armature");
    init.damping = pt->get<uint>("environment.damping");
    init.approx = pt->get<uint>("environment.approx");
    init.mu = pt->get<double>("environment.mu");
    init.mu2 = pt->get<double>("environment.mu2");
    init.soft_cfm = pt->get<double>("environment.soft_cfm");
    init.slip1 = pt->get<double>("environment.slip1");
    init.slip2 = pt->get<double>("environment.slip2");
    init.soft_erp = pt->get<double>("environment.soft_erp");
    init.bounce = pt->get<double>("environment.bounce");
    visible     = vm->count("view");

    if (visible)
      instance = new HalfCheetahWorldView("data/textures", init);
    else
      instance = new HalfCheetahWorld(init);
  }

  void _apply(const std::vector<double>& actuators) {
    instance->step(actuators);
    
    double ctrl_cost = 0;
    for (auto a : actuators)
      ctrl_cost += a*a;
    ctrl_cost = 1e-1 * 0.5 * ctrl_cost;
    
    double run_cost = -1.f * instance->torso_velocity();
    reward = ctrl_cost + run_cost;
    reward = -reward;
    
    if(perceptions()[1] < 0.2 || perceptions()[1] > 0.85)
      reward = -100000;
  }

  void _reset_episode(bool learning) override {
    std::vector<double> given_stoch={0,0};
    if(!learning)
      given_stoch.clear();
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
  void _reset_episode_choose(const std::vector<double>& given_stoch) override {
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
    void _next_instance(bool learning) override {
    std::vector<double> given_stoch={0,0};
    if(!learning)
      given_stoch.clear();
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
  void _next_instance_choose(const std::vector<double>& given_stoch) override {
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }

 private:
  bool visible = false;
  HalfCheetahWorld* instance;
  double reward; 
  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
