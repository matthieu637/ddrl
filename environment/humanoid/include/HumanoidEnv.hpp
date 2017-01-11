#ifndef HALTCHEETAHENV_H
#define HALTCHEETAHENV_H

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

  const std::vector<double>& perceptions() const {
    return instance->state();
  }

  double performance() const {
    return instance->performance();
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
    humanoid_physics init;
    init.apply_armature = pt->get<bool>("environment.apply_armature");
    init.damping = pt->get<uint>("environment.damping");
    init.approx = pt->get<uint>("environment.approx");
    init.control = pt->get<uint>("environment.control");
    init.reward = pt->get<uint>("environment.reward");
    init.mu = pt->get<double>("environment.mu");
    init.mu2 = pt->get<double>("environment.mu2");
    init.soft_cfm = pt->get<double>("environment.soft_cfm");
    init.slip1 = pt->get<double>("environment.slip1");
    init.slip2 = pt->get<double>("environment.slip2");
    init.soft_erp = pt->get<double>("environment.soft_erp");
    init.bounce = pt->get<double>("environment.bounce");
    visible     = vm->count("view");
    
    init.predev = 0;
    try {
      init.predev = pt->get<uint>("environment.predev");
    } catch(boost::exception const& ) {
    }
    init.from_predev = 0;
    try {
      init.from_predev = pt->get<uint>("environment.from_predev");
    } catch(boost::exception const& ) {
    }
    
    ASSERT(init.predev == 0 || init.from_predev ==0, "for now only one dev");

    if (visible)
      instance = new HumanoidWorldView("data/textures", init);
    else
      instance = new HumanoidWorld(init);
  }

  void _apply(const std::vector<double>& actuators) {
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
