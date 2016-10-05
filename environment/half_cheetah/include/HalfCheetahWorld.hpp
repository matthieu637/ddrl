#ifndef HALTCHEETAHWORLD_HPP
#define HALTCHEETAHWORLD_HPP

#include <vector>
#include "ode/ode.h"

#include "bib/Assert.hpp"
#include "ODEObject.hpp"
#include "ODEFactory.hpp"

#define GRAVITY -9.81
#define DAMPING 0.1
#define ARMATURE 0.1
#define WORLD_STEP 0.01

struct hcheetah_physics{
  bool apply_armature;
  uint approx;
  uint damping;
  uint control;
  uint reward;
  double mu;
  double mu2;
  double soft_cfm;
  double slip1;
  double slip2;
  double soft_erp;
  double bounce;
};

class HalfCheetahWorld {
 public:
  HalfCheetahWorld(const hcheetah_physics phy);
  virtual ~HalfCheetahWorld();

  void resetPositions(std::vector<double> &, const std::vector<double>& given_stoch);
  
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

  std::vector<double> internal_state;
};

struct nearCallbackDataHalfCheetah {
  HalfCheetahWorld *inst;
};

#endif  // ADVANCEDACROBOTWORLD_HPP
