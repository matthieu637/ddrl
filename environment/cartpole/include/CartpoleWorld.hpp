#ifndef CARTPOLEWORLD_HPP
#define CARTPOLEWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define POLE_LENGTH 0.5
//sqrt(0,1÷(0,5×1062)) so mass = 0.1
#define POLE_LARGER 0.013723116

#define CART_LARGER 0.098014838
//((1÷1062)^(1÷3)) so mass = 1

#define MAX_SLIDER_POSITON 2.4
#define MAX_HINGE_ANGLE 0.523598776 //PI/6

#define GRAVITY -9.81
#define BONE_DENSITY 1062  // Average human body density
#define MAX_TORQUE_SLIDER 10
#define WORLD_STEP 0.01

class CartpoleWorld {
 public:
  CartpoleWorld(bool add_time_in_state=false, bool normalization=false);
  virtual ~CartpoleWorld();

  void resetPositions(std::vector<double> &, const std::vector<double>& given_stoch);
  
  bool final_state() const;
  
  bool goal_state() const;

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
  
 protected:
  std::vector<dJointID> joints;

  std::vector<double> internal_state;

 private:
  bool add_time_in_state;
  bool normalization;

  static const std::vector<double> NORMALIZED_VEC;
};

struct nearCallbackDataCartpole {
  CartpoleWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
