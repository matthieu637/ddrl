#ifndef ADVANCEDACROBOTENV_H
#define ADVANCEDACROBOTENV_H

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
    delete instance;
  }

  const std::vector<double>& perceptions() const {
    return instance->state();
  }

  double performance() const {
    return 1.f;
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

    bool normalization = false;
    try {
      normalization = properties->get<bool>("environment.normalization");
    } catch(boost::exception const& ) {
      LOG_INFO("doest not normalize");
    }

    if (visible)
      instance = new CartpoleWorldView("data/textures", add_time_in_state, normalization);
    else
      instance = new CartpoleWorld(add_time_in_state, normalization);
  }

  void _apply(const std::vector<double>& actuators) {
    instance->step(actuators, current_step, max_step_per_instance);
  }

  void _next_instance() {
    instance->resetPositions();
  }

 private:
  bool visible = false;
  CartpoleWorld* instance;

  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
