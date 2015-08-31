#ifndef CARTPOLEWORLD_HPP
#define CARTPOLEWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define BONE_LENGTH 0.2
#define BONE_LARGER 0.04
#define STARTING_Z 0.5

#define GRAVITY -9.81
#define BONE_DENSITY 1062  // Average human body density
#define MAX_TORQUE_SLIDER 1.5
#define WORLD_STEP 0.005

class CartpoleWorld {
 public:
  CartpoleWorld(bool add_time_in_state=false, bool normalization=false);
  virtual ~CartpoleWorld();

  void resetPositions();
  
  bool final_state() const;

  virtual void step(const std::vector<double> &motors, uint current_step, uint max_step_per_instance);
  const std::vector<double> &state() const;
  unsigned int activated_motors() const;

 protected:
  void createWorld();
  void update_state(uint current_step, uint max_step_per_instance);

 public:
  ODEWorld odeworld;
  std::vector<ODEObject *> bones;
  dGeomID ground;
  bool touchGround;
  
 protected:
  std::vector<dJointID> joints;

  std::vector<double> internal_state;

 private:
  bool normalization;

  static const std::vector<double> NORMALIZED_VEC;
};

struct nearCallbackDataCartpole {
  CartpoleWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
