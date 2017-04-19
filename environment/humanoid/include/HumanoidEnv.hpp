#ifndef HUMANOIDENV_H
#define HUMANOIDENV_H

#include <string>
#include <vector>

#include "bib/IniParser.hpp"
#include "arch/AEnvironment.hpp"
#include "HumanoidWorld.hpp"
#include "HumanoidWorldView.hpp"

class HumanoidEnv : public arch::AEnvironment<> {
 public:
  HumanoidEnv() {
    ODEFactory::getInstance();
    instance = nullptr;
  }

  ~HumanoidEnv() {
    delete instance;
  }

  const std::vector<double>& perceptions() const override {
    return instance->state();
  }

  double performance() const override {
    return instance->performance();
  }

  bool final_state() const override {
    return instance->final_state();
  }

  unsigned int number_of_actuators() const override {
    return instance->activated_motors();
  }

  unsigned int number_of_sensors() const override {
    return instance->state().size();
  }

 private:
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* vm) override {
    humanoid_physics init;
    init.apply_armature = pt->get<bool>("environment.apply_armature");
    init.damping = pt->get<uint>("environment.damping");
    init.approx = pt->get<uint>("environment.approx");
    init.control = pt->get<uint>("environment.control");
    init.mu = pt->get<double>("environment.mu");
    init.mu2 = pt->get<double>("environment.mu2");
    init.soft_cfm = pt->get<double>("environment.soft_cfm");
    init.slip1 = pt->get<double>("environment.slip1");
    init.slip2 = pt->get<double>("environment.slip2");
    init.soft_erp = pt->get<double>("environment.soft_erp");
    init.bounce = pt->get<double>("environment.bounce");
    init.additional_sensors = pt->get<bool>("environment.additional_sensors");
    init.reward_scale_lvc = pt->get<double>("environment.reward_scale_lvc");
    init.reward_penalty_dead = pt->get<double>("environment.reward_penalty_dead");
    visible     = vm->count("view");
    
    
    if (visible)
      instance = new HumanoidWorldView("data/textures", init);
    else
      instance = new HumanoidWorld(init);
  }

  void _apply(const std::vector<double>& actuators) override {
    instance->step(actuators);
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
  HumanoidWorld* instance;
  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
