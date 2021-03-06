#ifndef HALTCHEETAHWORLD_HPP
#define HALTCHEETAHWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define GRAVITY -9.81
#define WORLD_STEP 0.01
// instead of frame skip, reduce precision
// mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
// #define WORLD_STEP 0.0125
// #define FRAME_SKIP 4
#define FRAME_SKIP 5

struct hcheetah_physics{
  bool apply_armature;
  uint damping;
  uint control;
  bool pd_controller;
  double soft_cfm;
  double bounce;
  double bounce_vel;
  uint predev;
  uint from_predev;
//   PREDEV: 1 -> two low joint
//           10 -> two middle joints
//   +0 (1/10) -> PD controller init pos (sensors removed)
//   +1 (2/11) -> PD controller init pos (keep sensors)
//   +2 (3/12) -> PD controller init pos (keep sensors but setted to 0)
  bool lower_rigid; //computed
  bool higher_rigid;
};

class HalfCheetahWorld {
 public:
  HalfCheetahWorld(const hcheetah_physics phy);
  virtual ~HalfCheetahWorld();

  virtual void resetPositions(std::vector<double> &, const std::vector<double>& given_stoch);
  
  bool final_state() const;
  
  double performance() const;

  virtual void step(const std::vector<double> &motors);
  virtual void step_core(const std::vector<double> &motors);
  const std::vector<double> &state() const;
  unsigned int activated_motors() const;

 protected:
  virtual void createWorld();
  void update_state();
  void apply_armature(dMass* m, double k);
  void apply_damping(dBodyID body, double v);

 public:
  ODEWorld odeworld;
  std::vector<ODEObject *> bones;
  dGeomID ground;
  hcheetah_physics phy;
  dContact contact[2];
  bool head_touch, fknee_touch, bknee_touch;
  
 protected:
  std::vector<dJointID> joints;
  double penalty;
  double reward;
  double pos_before;

  std::vector<double> internal_state;
};

struct nearCallbackDataHalfCheetah {
  HalfCheetahWorld *inst;
};

void nearCallbackHalfCheetah(void* data, dGeomID o1, dGeomID o2);

#endif  // ADVANCEDACROBOTWORLD_HPP
