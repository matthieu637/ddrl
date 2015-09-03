#include "CartpoleWorld.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include "ode/ode.h"

#include "bib/Utils.hpp"
#include "ODEFactory.hpp"


CartpoleWorld::CartpoleWorld(bool _add_time_in_state, bool _normalization)
  : odeworld(ODEFactory::getInstance()->createWorld()), add_time_in_state(_add_time_in_state), normalization(_normalization) {

  dWorldSetGravity(odeworld.world_id, 0, 0.0, GRAVITY);

  createWorld();

  internal_state.push_back(dJointGetSliderPosition(joints[0]));
  internal_state.push_back(dJointGetSliderPositionRate(joints[0]));
  internal_state.push_back(dJointGetHingeAngle(joints[1]));
  internal_state.push_back(dJointGetHingeAngleRate(joints[1]));

  if(add_time_in_state)
    internal_state.push_back(0.);
  
  resetPositions();
}

CartpoleWorld::~CartpoleWorld() {
  for (ODEObject * o : bones) {
    dGeomDestroy(o->getGeom());
    dBodyDestroy(o->getID());
    delete o;
  }

  dGeomDestroy(ground);

  ODEFactory::getInstance()->destroyWorld(odeworld);
}

void CartpoleWorld::createWorld() {
  double bone_length = BONE_LENGTH;
  double bone_larger = BONE_LARGER;
  double starting_z = bone_larger / 2.f;

  ground = ODEFactory::getInstance()->createGround(odeworld);
// 
//   //  Create the first fixed bone with it's joint
  ODEObject* first_bone = ODEFactory::getInstance()->createBox(
                            odeworld, 0., 0, bone_larger / 2.f, bone_length/2.f, bone_larger, bone_larger / 2.f,
                            BONE_DENSITY, true );                
  bones.push_back(first_bone);

  dJointID first_slider = dJointCreateSlider(odeworld.world_id, nullptr);
  dJointAttach(first_slider, 0, first_bone->getID());
  dJointSetSliderAxis(first_slider, 1, 0, 0);
  dJointSetSliderParam(first_slider, dParamHiStop, MAX_SLIDER_POSITON);
  dJointSetSliderParam(first_slider, dParamLoStop, -MAX_SLIDER_POSITON);
  joints.push_back(first_slider);
  
  ODEObject* second_bone = ODEFactory::getInstance()->createBox(
                            odeworld, 0., 0, bone_larger / 2.f + bone_length / 2.f, bone_larger / 3.f, bone_larger / 3.f, bone_length,
                            BONE_DENSITY, true );
  bones.push_back(second_bone);
  
  dJointID first_hinge = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(first_hinge, first_bone->getID(), second_bone->getID());
//   dJointSetHingeAnchor(first_hinge, 0, 0, starting_z + bone_length / 2.);
  dJointSetHingeAxis(first_hinge, 0, 1, 0);
  joints.push_back(first_hinge);

}

void nearCallbackCartpole(void* data, dGeomID o1, dGeomID o2) {
  int n;
  nearCallbackDataCartpole* d = reinterpret_cast<nearCallbackDataCartpole*>(data);
  CartpoleWorld* inst = d->inst;

  // exit without doing anything if the two bodies are connected by a joint
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnectedExcluding(b1, b2, dJointTypeContact)) 
    return;

  const int N = 10;
  dContact contact[N];
  n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
  if (n > 0) {
    if( (o1 == inst->bones[1]->getGeom() && o2 == inst->ground) ||
        (o2 == inst->bones[1]->getGeom() && o1 == inst->ground) ){
      inst->touchGround = true;
    }
    
    for (int i = 0; i < n; i++) {
      contact[i].surface.mode = dContactSlip1 | dContactSlip2 | dContactSoftERP | dContactSoftCFM | dContactApprox1;
      contact[i].surface.mu = dInfinity;
      contact[i].surface.slip1 = 0.5;
      contact[i].surface.slip2 = 0.5;
      contact[i].surface.soft_erp = 0.95;
      contact[i].surface.soft_cfm = 0.5;

      dJointID c = dJointCreateContact(inst->odeworld.world_id, inst->odeworld.contactgroup, &contact[i]);
      dJointAttach(c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
    }
  }
}

void CartpoleWorld::step(const vector<double>& motors, uint current_step, uint max_step_per_instance) {
  // No collision in this world

  nearCallbackDataCartpole d = {this};
  dSpaceCollide(odeworld.space_id, &d, &nearCallbackCartpole);

  unsigned int begin_index = 0;
  double force = 0.f;
  
  force = bib::Utils::transform(motors[begin_index++], -1, 1, -MAX_TORQUE_SLIDER, MAX_TORQUE_SLIDER);
  dJointAddSliderForce(joints[0], force);

// 
//   for (unsigned int i = 1; i < actuators.size(); i++)
//     if (actuators[i]) {
//       if (types[i - 1] == HINGE)
//         force = bib::Utils::transform(motors[begin_index++], -1, 1, -MAX_TORQUE_HINGE, MAX_TORQUE_HINGE);
//       else
//         force = bib::Utils::transform(motors[begin_index++], -1, 1, -MAX_TORQUE_SLIDER, MAX_TORQUE_SLIDER);
// 
//       dJointAddHingeTorque(joints[i], force);
//     }
// 
//   ASSERT(begin_index == motors.size(),
//          "wrong number of motors " << begin_index << " " << motors.size() << " " << actuators.size());

  Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
  dWorldStep(odeworld.world_id, WORLD_STEP);
  lock.release();

  dJointGroupEmpty(odeworld.contactgroup);

  update_state(current_step, max_step_per_instance);
}

const std::vector<double> CartpoleWorld::NORMALIZED_VEC({28,62,71});

void CartpoleWorld::update_state(uint current_step, uint max_step_per_instance) {
//   uint begin_index = 0;
// 
//   if(normalization)
//     internal_state[begin_index++] = bib::Utils::transform(dJointGetHingeAngle(joints[0]), -M_PI, M_PI, -1, 1);
//   else
//     internal_state[begin_index++] = dJointGetHingeAngle(joints[0]);
// 
//   if(normalization)
//     internal_state[begin_index++] = bib::Utils::transform(dJointGetHingeAngleRate(joints[0]), -NORMALIZED_VEC[0],
//                                     NORMALIZED_VEC[0], -1, 1);
//   else
//     internal_state[begin_index++] = dJointGetHingeAngleRate(joints[0]);
// 
//   for (unsigned int i = 0; i < types.size(); i++)
//     if (types[i] == HINGE) {
//       if(normalization)
//         internal_state[begin_index++] = bib::Utils::transform(dJointGetHingeAngle(joints[i + 1]), -M_PI, M_PI, -1, 1);
//       else
//         internal_state[begin_index++] = dJointGetHingeAngle(joints[i + 1]);
// 
//       if(normalization && i+1 < NORMALIZED_VEC.size())
//         internal_state[begin_index++] = bib::Utils::transform(dJointGetHingeAngleRate(joints[i + 1]), -NORMALIZED_VEC[i + 1],
//                                         NORMALIZED_VEC[i + 1], -1, 1);
//       else
//         internal_state[begin_index++] = dJointGetHingeAngleRate(joints[i + 1]);
//     } else {
//       internal_state[begin_index++] = dJointGetSliderPosition(joints[i + 1]);
//       internal_state[begin_index++] = dJointGetSliderPositionRate(joints[i + 1]);
//     }
// 
//   internal_state[begin_index] = bib::Utils::transform(current_step, 0, max_step_per_instance, -1.f, 1.f);
  uint begin_index = 0;
  
  internal_state[begin_index++] = dJointGetSliderPosition(joints[0]);
  internal_state[begin_index++] = dJointGetSliderPositionRate(joints[0]);
  internal_state[begin_index++] = dJointGetHingeAngle(joints[1]);
  internal_state[begin_index++] = dJointGetHingeAngleRate(joints[1]);
  
  if( fabs(dJointGetSliderPosition(joints[0])) >= MAX_SLIDER_POSITON )
    touchGround=true;
  
  if(add_time_in_state)
    internal_state[begin_index] = bib::Utils::transform(current_step, 0, max_step_per_instance, -1.f, 1.f);
}

const std::vector<double>& CartpoleWorld::state() const {
  return internal_state;
}

unsigned int CartpoleWorld::activated_motors() const {
  return 1;
}

bool CartpoleWorld::final_state() const {
  return touchGround;
}

void CartpoleWorld::resetPositions() {
  double bone_length = BONE_LENGTH;
  double bone_larger = BONE_LARGER;
  double starting_z = bone_larger / 2.f + bone_length / 2.f;

  dJointAddSliderForce(joints[0], 0);

  dMatrix3 R;
  dRFromEulerAngles(R, 0, 0, 0);

  unsigned int i = 0;
  for (ODEObject * o : bones) {
    dBodySetRotation(o->getID(), R);
    dBodySetForce(o->getID(), 0, 0, 0);
    dBodySetLinearVel(o->getID(), 0, 0, 0);
    dBodySetAngularVel(o->getID(), 0, 0, 0);
  }
  
  dBodySetPosition(bones[0]->getID(), 0, 0, bone_larger / 2.f);
  dBodySetPosition(bones[1]->getID(), 0, 0, starting_z);

  touchGround = false;
  
  double r = bib::Utils::rand01();
  r = r <= 0.5 ? r - 1.f : r;
  r *= MAX_TORQUE_SLIDER;
  dJointAddSliderForce(joints[0], r);
    
  for (uint n=0; n < 10; n++){    
    Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
    dWorldStep(odeworld.world_id, WORLD_STEP);
    lock.release();
  }
  
  dJointAddSliderForce(joints[0], 0);

  dJointGroupEmpty(odeworld.contactgroup);

  update_state(0, 1);
}
