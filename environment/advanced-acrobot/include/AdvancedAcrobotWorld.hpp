#ifndef ADVANCEDACROBOTWORLD_HPP
#define ADVANCEDACROBOTWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define BONE_LENGTH 1.f
//sqrt(1÷1062) so mass=1
#define BONE_LARGER 0.030685821
//#define BONE_LARGER 0.021698152
#define STARTING_Z 0.5
#define INERTIA 1.f

#define GRAVITY -9.81
#define BONE_DENSITY 1062  // Average human body density
#define MAX_TORQUE_HINGE 1.5
#define MAX_TORQUE_SLIDER 5 // not used
#define WORLD_STEP 0.01

enum bone_joint { HINGE, SLIDER };

class AdvancedAcrobotWorld {
 public:
  AdvancedAcrobotWorld(const std::vector<bone_joint> &types = {HINGE, HINGE},
                       const std::vector<bool> &actuators = {false, false, true},
                       bool add_time_in_state=false, bool normalization=false, 
                       const std::vector<double>& normalized_vector = {}
                      );
  virtual ~AdvancedAcrobotWorld();

  void resetPositions();

  virtual void step(const std::vector<double> &motors, uint current_step, uint max_step_per_instance);
  const std::vector<double> &state() const;
  unsigned int activated_motors() const;
  double perf() const;

 protected:
  void createWorld(const std::vector<bone_joint> &);
  void update_state(uint current_step, uint max_step_per_instance);

 public:
  ODEWorld odeworld;

 protected:
  std::vector<bone_joint> types;
  std::vector<bool> actuators;

  std::vector<ODEObject *> bones;
  std::vector<dJointID> joints;

  std::vector<double> internal_state;

  dGeomID ground;

 private:
  bool goalBeenReached;
  bool goalFailed;
  unsigned int _activated_motors;
  bool add_time_in_state;
  bool normalization;

  std::vector<double> normalized_vector;
};

struct nearCallbackData {
  AdvancedAcrobotWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
