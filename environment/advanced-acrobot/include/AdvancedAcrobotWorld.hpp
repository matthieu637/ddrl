#ifndef ADVANCEDACROBOTWORLD_HPP
#define ADVANCEDACROBOTWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define BONE_LENGTH 0.3
#define BONE_LARGER 0.02
#define STARTING_Z 0.5

#define GRAVITY -9.81
#define BONE_DENSITY 1062  // Average human body density
#define MAX_TORQUE_HINGE 0.125
#define MAX_TORQUE_SLIDER 5
#define WORLD_STEP 0.01

enum bone_joint { HINGE, SLIDER };

class AdvancedAcrobotWorld {
 public:
  AdvancedAcrobotWorld(const std::vector<bone_joint> &types = {HINGE, HINGE},
                       const std::vector<bool> &actuators = {false, false, true},
                       bool add_time_in_state=false, bool normalization=false
                      );
  virtual ~AdvancedAcrobotWorld();

  void resetPositions();

  virtual void step(const std::vector<float> &motors, uint current_step, uint max_step_per_instance);
  const std::vector<float> &state() const;
  unsigned int activated_motors() const;
  float perf() const;

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

  std::vector<float> internal_state;

  dGeomID ground;

 private:
  bool goalBeenReached;
  bool goalFailed;
  unsigned int _activated_motors;
  bool normalization;
  
  static const std::vector<float> NORMALIZED_VEC;
};

struct nearCallbackData {
  AdvancedAcrobotWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
