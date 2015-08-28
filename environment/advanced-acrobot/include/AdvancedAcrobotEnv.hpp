#ifndef ADVANCEDACROBOTENV_H
#define ADVANCEDACROBOTENV_H

#include <string>
#include <vector>

#include "bib/IniParser.hpp"
#include "arch/AEnvironment.hpp"
#include "AdvancedAcrobotWorld.hpp"
#include "AdvancedAcrobotWorldView.hpp"

std::istream& operator>>(std::istream& istream, bone_joint& v);

class ProblemDefinition {
 public:
  virtual ~ProblemDefinition() {}

 public:
  virtual double performance(AdvancedAcrobotWorld*) = 0;
  virtual bool still_running(AdvancedAcrobotWorld*) const {
    return true;
  }
  virtual void setEnv(arch::AEnvironment<> *) {

  }
  virtual void reset() { }
};

class KeepHigh : public ProblemDefinition {
 public:
  double performance(AdvancedAcrobotWorld* instance) {
    return instance->perf();
  }
};

class ReachLimitPoorInformed : public ProblemDefinition {
 public:
  double performance(AdvancedAcrobotWorld* instance) {
    if (instance->perf() > 0.99f)
      return 1.;
    else
      return 0.f;
  }
  bool still_running(AdvancedAcrobotWorld* instance) const {
    return instance->perf() <= 0.99f;
  }
};

class ReachLimitPoorInformedNoGamma : public ProblemDefinition {
 public:
  double performance(AdvancedAcrobotWorld* instance) {
    if (instance->perf() > 0.99d)
      return 1.00d;
    else
      return -1.00d;
  }
  bool still_running(AdvancedAcrobotWorld* instance) const {
    return instance->perf() <= 0.99f;
  }
};

class ReachLimitPoorInformedMax : public ProblemDefinition {
 public:
  double performance(AdvancedAcrobotWorld* instance) {
    if(current_max < instance->perf())
      current_max = instance->perf();

    if (instance->perf() > 0.99f)
      return 1.;
    else if(env->running())
      return 0.;
    else
      return current_max;
  }

  bool still_running(AdvancedAcrobotWorld* instance) const {
    return instance->perf() <= 0.99f;
  }

  void setEnv(arch::AEnvironment<> * _env) {
    env = _env;
  }

  void reset() {
    current_max = 0;
  }
 protected:
  double current_max = 0;
  arch::AEnvironment<> * env;
};

class ReachLimitWellInformed : public ProblemDefinition {
 public:
  double performance(AdvancedAcrobotWorld* instance) {
    if (instance->perf() > 0.99f)
      return 1.;
    else
      return instance->perf() * 0.01;
  }
  bool still_running(AdvancedAcrobotWorld* instance) const {
    return instance->perf() <= 0.99f;
  }
};

ProblemDefinition* str2prob(const std::string& s);

class AdvancedAcrobotEnv : public arch::AEnvironment<> {
 public:
  AdvancedAcrobotEnv() {
    ODEFactory::getInstance();
    bones = nullptr;
    actuators = nullptr;
    instance = nullptr;
    problem = nullptr;
  }

  ~AdvancedAcrobotEnv() {
    delete bones;
    delete actuators;
    delete problem;
    delete instance;
  }

  const std::vector<double>& perceptions() const {
    return instance->state();
  }

  double performance() const {
    return problem->performance(instance);
  }

  bool final_state() const {
    if (!visible)
      return !problem->still_running(instance);
    if (!problem->still_running(instance))
      LOG_INFO("goal reached but continue simulation [--view]");
    return false;
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
    bones       = bib::to_array<bone_joint>(properties->get<std::string>("environment.bones"));
    actuators   = bib::to_array<bool>(properties->get<std::string>("environment.actuators"));
    problem     = str2prob(properties->get<std::string>("environment.problem"));
    problem->setEnv(this);

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
      instance = new AdvancedAcrobotWorldView("data/textures", *bones, *actuators, add_time_in_state, normalization);
    else
      instance = new AdvancedAcrobotWorld(*bones, *actuators, add_time_in_state, normalization);
  }

  void _apply(const std::vector<double>& actuators) {
    instance->step(actuators, current_step, max_step_per_instance);
  }

  void _next_instance() {
    instance->resetPositions();
    problem->reset();
  }

 private:
  bool visible = false;
  std::vector<bone_joint>* bones;
  std::vector<bool>* actuators;
  AdvancedAcrobotWorld* instance;
  ProblemDefinition* problem;

  std::vector<double> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
