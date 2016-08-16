#ifndef CARTPOLEENV_H
#define CARTPOLEENV_H

#include <string>
#include <vector>

#include "bib/IniParser.hpp"
#include "arch/AEnvironment.hpp"
#include "CartpoleWorld.hpp"
#include "CartpoleWorldView.hpp"

class CartpoleEnv : public arch::AEnvironment<> {
 public:
  CartpoleEnv() {
    ODEFactory::getInstance();
    instance = nullptr;
  }

  ~CartpoleEnv() {
    delete normalized_vector;
    delete instance;
  }

  const std::vector<double>& perceptions() const {
    return instance->state();
  }

  double performance() const {
    if(instance->goal_state())
      return 0;
    else if(final_state()){
      return -2.f*(500.f - current_step);
    }
    
    return -1.f;
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
  void _unique_invoke(boost::property_tree::ptree* properties, boost::program_options::variables_map* vm) {
    visible     = vm->count("view");

    bool add_time_in_state = false;
    try {
      add_time_in_state = properties->get<bool>("environment.add_time_in_state");
    } catch(boost::exception const& ) {
      LOG_INFO("doest not add time in state");
    }
    
    try {
      normalization = properties->get<bool>("environment.normalization");
    } catch(boost::exception const& ) {
      LOG_INFO("doest not normalize");
    }
    
    if(normalization)
      normalized_vector = bib::to_array<double>(properties->get<std::string>("environment.normalized_vector"));
    else
      normalized_vector = new std::vector<double>;

    if (visible)
      instance = new CartpoleWorldView("data/textures", add_time_in_state, normalization, *normalized_vector);
    else
      instance = new CartpoleWorld(add_time_in_state, normalization, *normalized_vector);
  }

  void _apply(const std::vector<double>& actuators) {
    instance->step(actuators, current_step, max_step_per_instance);
  }

  void _reset_episode() override {
    std::vector<double> given_stoch;
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
  void _reset_episode_choose(const std::vector<double>& given_stoch) override {
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
  void _next_instance() override {
    std::vector<double> given_stoch;
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }
  
  void _next_instance_choose(const std::vector<double>& given_stoch) override {
    instance->resetPositions(first_state_stochasticity, given_stoch);
  }

 private:
  bool visible = false;
  bool normalization = false;
  CartpoleWorld* instance;
  std::vector<double>* normalized_vector;

  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
