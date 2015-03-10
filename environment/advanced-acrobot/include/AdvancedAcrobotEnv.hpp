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
  virtual float performance(AdvancedAcrobotWorld*) const = 0;
  virtual bool still_running(AdvancedAcrobotWorld*) const {
    return true;
  }
};

class KeepHigh : public ProblemDefinition {
 public:
  float performance(AdvancedAcrobotWorld* instance) const {
    return instance->perf();
  }
};

class ReachLimitPoorInformed : public ProblemDefinition {
 public:
  float performance(AdvancedAcrobotWorld* instance) const {
    if (instance->perf() > 0.99f)
      return 1.;
    else
      return 0.f;
  }
  bool still_running(AdvancedAcrobotWorld* instance) const {
    return instance->perf() <= 0.99f;
  }
};

class ReachLimitWellInformed : public ProblemDefinition {
 public:
  float performance(AdvancedAcrobotWorld* instance) const {
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
    ODEFactory::endInstance();
  }

  const std::vector<float>& perceptions() const {
    return instance->state();
  }

  float performance() const {
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

    if (visible)
      instance = new AdvancedAcrobotWorldView("data/textures", *bones, *actuators);
    else
      instance = new AdvancedAcrobotWorld(*bones, *actuators);
  }

  void _apply(const std::vector<float>& actuators) {
    instance->step(actuators);
  }

  void _next_instance() {
    instance->resetPositions();
  }

 private:
  bool visible = false;
  std::vector<bone_joint>* bones;
  std::vector<bool>* actuators;
  AdvancedAcrobotWorld* instance;
  ProblemDefinition* problem;

  std::vector<float> internal_state;
};

#endif  // ADVANCEDACROBOTENV_H
