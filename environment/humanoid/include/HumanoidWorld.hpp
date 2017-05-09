#ifndef HUMANOIDWORLD_HPP
#define HUMANOIDWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define GRAVITY -9.81
#define DAMPING 0.1
#define ARMATURE 0.1

// <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
#define WORLD_STEP 0.003

// mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
#define FRAME_SKIP 5

// in ODE the agent can found static position and waits
#define ALIVE_BONUS 1.0

struct humanoid_physics{
  bool apply_armature;
  uint approx_ground;
  uint damping;
  uint control;
  double soft_cfm;
  double slip1;
  double slip2;
  double soft_erp;
  double bounce_ground;
  double bounce_vel;
  bool additional_sensors;
  double reward_scale_lvc;
  double reward_penalty_dead;
  bool reapply_motors;
  bool reupdate_state;
};

class HumanoidWorld {
 public:
  HumanoidWorld(const humanoid_physics phy);
  virtual ~HumanoidWorld();

  virtual void resetPositions(std::vector<double> &, const std::vector<double>& given_stoch);
  
  bool final_state() const;
  
  double performance() const;

  virtual void step(const std::vector<double> &motors);
  const std::vector<double> &state() const;
  unsigned int activated_motors() const;
  double mass_center();

 protected:
  void createWorld();
  void update_state(bool updateReward);
  void apply_armature(dMass* m, double k);
  void apply_damping(dBodyID body, double v);
//   void copy_inertia(dMass* m, uint i);

 public:
  ODEWorld odeworld;
  std::vector<ODEObject *> bones;
  dGeomID ground;
  humanoid_physics phy;
  dContact contact_ground[2];
  dContact contact_body[2];
  
//   static const std::vector<double> mujoco_inertia;
  
 protected:
  std::vector<dJointID> joints;
  double reward;
  std::vector<double> internal_state;
  std::vector<double> body_mass;
  std::vector<double> qfrc_actuator;
  std::vector<double> gears;
  double mass_sum;
  double pos_before;
};

struct nearCallbackDataHumanoid {
  HumanoidWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
