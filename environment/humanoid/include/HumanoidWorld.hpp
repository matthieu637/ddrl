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

#define ALIVE_BONUS 5.0

struct humanoid_physics{
  bool apply_armature;
  uint approx;
  uint damping;
  uint control;
  double mu;
  double mu2;
  double soft_cfm;
  double slip1;
  double slip2;
  double soft_erp;
  double bounce;
  bool additional_sensors;
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

 protected:
  void createWorld();
  void update_state();
  void apply_armature(dMass* m, double k);
  void apply_damping(dBodyID body, double v);
//   void copy_inertia(dMass* m, uint i);
  double mass_center();

 public:
  ODEWorld odeworld;
  std::vector<ODEObject *> bones;
  dGeomID ground;
  humanoid_physics phy;
  dContact contact[2];
  
//   static const std::vector<double> mujoco_inertia;
  
 protected:
  std::vector<dJointID> joints;
  double reward;
  std::vector<double> internal_state;
  std::vector<double> body_mass;
  std::vector<double> qfrc_actuator;
  double mass_sum;
  double pos_before;
};

struct nearCallbackDataHumanoid {
  HumanoidWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
