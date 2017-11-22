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
    hcheetah_physics init;
    init.apply_armature = pt->get<bool>("environment.apply_armature");
    init.damping = pt->get<uint>("environment.damping");
    init.control = pt->get<uint>("environment.control");
    init.soft_cfm = pt->get<double>("environment.soft_cfm");
    init.bounce = pt->get<double>("environment.bounce");
    init.bounce_vel = 0;    
    if (init.bounce >= 0.0000f)
      init.bounce_vel = pt->get<double>("environment.bounce_vel");
    bool visible = vm->count("view");
    bool capture = vm->count("capture");
    
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
    
    init.pd_controller = true;
    try {
      init.pd_controller = pt->get<bool>("environment.pd_controller");
    } catch(boost::exception const& ) {
    }
    
    ASSERT(init.predev == 0 || init.from_predev ==0, "for now only one dev");
    
    init.lower_rigid = init.control == 1 && init.predev >= 1 && init.predev <= 9;
    init.higher_rigid = init.control == 1 && init.predev >= 10;
    
    if((init.lower_rigid || init.higher_rigid) && (init.predev == 3 || init.predev == 12)){
      LOG_ERROR("control = 1 with predev 3/12 is the same as control = 1 with 2/11");
      LOG_ERROR("3/12 forces sensors to 0 when they are defined (control=0)");
      exit(1);
    }

    if (visible)
      instance = new HalfCheetahWorldView("data/textures", init, capture);
    else
      instance = new HalfCheetahWorld(init);
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
  HalfCheetahWorld* instance;
  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
