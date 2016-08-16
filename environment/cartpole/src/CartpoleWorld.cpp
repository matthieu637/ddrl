#include "CartpoleWorld.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include "ode/ode.h"

#include "bib/Utils.hpp"
#include "ODEFactory.hpp"


CartpoleWorld::CartpoleWorld(bool _add_time_in_state, bool _normalization, const std::vector<double>& _normalized_vector)
  : odeworld(ODEFactory::getInstance()->createWorld()), add_time_in_state(_add_time_in_state), 
  normalization(_normalization), normalized_vector(_normalized_vector) {

  dWorldSetGravity(odeworld.world_id, 0, 0.0, GRAVITY);

  createWorld();

  internal_state.push_back(dJointGetSliderPosition(joints[0]));
  internal_state.push_back(dJointGetSliderPositionRate(joints[0]));
  internal_state.push_back(dJointGetHingeAngle(joints[1]));
  internal_state.push_back(dJointGetHingeAngleRate(joints[1]));

  if(add_time_in_state)
    internal_state.push_back(0.);
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
  ground = ODEFactory::getInstance()->createGround(odeworld);
// 
//   //  Create the first fixed bone with it's joint
  ODEObject* first_bone = ODEFactory::getInstance()->createBox(
                            odeworld, 0., 0, CART_LARGER/2.f, CART_LARGER, CART_LARGER, CART_LARGER,
                            BONE_DENSITY, true, INERTIA);                
  bones.push_back(first_bone);

  dJointID first_slider = dJointCreateSlider(odeworld.world_id, nullptr);
  dJointAttach(first_slider, 0, first_bone->getID());
  dJointSetSliderAxis(first_slider, 1, 0, 0);
  dJointSetSliderParam(first_slider, dParamHiStop, MAX_SLIDER_POSITON);
  dJointSetSliderParam(first_slider, dParamLoStop, -MAX_SLIDER_POSITON);
  //simulate friction:
//   dJointSetSliderParam(first_slider, dParamFMax, 0.9);
//   dJointSetSliderParam(first_slider, dParamVel, 0);
//   dJointSetSliderParam(first_slider, dParamFMax1, 0.9);
//   dJointSetSliderParam(first_slider, dParamVel1, 0);
  joints.push_back(first_slider);
  
//   bib::Logger::PRINT_ELEMENTS(first_bone->getMass().I, 9, "cart inertia : ");
  
  ODEObject* second_bone = ODEFactory::getInstance()->createBox(
                            odeworld, 0., 0, POLE_LENGTH/2.f+CART_LARGER/2.f, POLE_LARGER, POLE_LARGER, POLE_LENGTH,
                            BONE_DENSITY, true, INERTIA);
  bones.push_back(second_bone);
  
  dJointID first_hinge = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(first_hinge, first_bone->getID(), second_bone->getID());
  //dJointSetHingeAnchor(first_hinge, 0, 0, POLE_LENGTH/2.f+CART_LARGER/2.f);
  dJointSetHingeAxis(first_hinge, 0, 1, 0);
  //simulate friction:
//   dJointSetHingeParam(first_hinge, dParamFMax, 0.000002);
//   dJointSetHingeParam(first_hinge, dParamVel, 0);
//   dJointSetHingeParam(first_hinge, dParamFMax2, 0.000002);
//   dJointSetHingeParam(first_hinge, dParamVel2, 0);
  joints.push_back(first_hinge);
  
//   bib::Logger::PRINT_ELEMENTS(second_bone->getMass().I, 9, "pole inertia : ");
}

// void nearCallbackCartpole(void* data, dGeomID o1, dGeomID o2) {
//   int n;
//   nearCallbackDataCartpole* d = reinterpret_cast<nearCallbackDataCartpole*>(data);
//   CartpoleWorld* inst = d->inst;
// 
//   // exit without doing anything if the two bodies are connected by a joint
//   dBodyID b1 = dGeomGetBody(o1);
//   dBodyID b2 = dGeomGetBody(o2);
//   if (b1 && b2 && dAreConnectedExcluding(b1, b2, dJointTypeContact)) 
//     return;
// 
//   const int N = 10;
//   dContact contact[N];
//   n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
//   if (n > 0) {
//     if( (o1 == inst->bones[1]->getGeom() && o2 == inst->ground) ||
//         (o2 == inst->bones[1]->getGeom() && o1 == inst->ground) ){
//       //inst->touchGround = true;
//     }
//     
//     for (int i = 0; i < n; i++) {
//       contact[i].surface.mode = dContactSlip1 | dContactSlip2 | dContactSoftERP | dContactSoftCFM | dContactApprox1;
//       contact[i].surface.mu = dInfinity;
//       contact[i].surface.slip1 = 0.5;
//       contact[i].surface.slip2 = 0.5;
//       contact[i].surface.soft_erp = 0.95;
//       contact[i].surface.soft_cfm = 0.5;
// 
//       dJointID c = dJointCreateContact(inst->odeworld.world_id, inst->odeworld.contactgroup, &contact[i]);
//       dJointAttach(c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
//     }
//   }
// }

void CartpoleWorld::step(const vector<double>& motors, uint current_step, uint max_step_per_instance) {
  // No collision in this world

  //nearCallbackDataCartpole d = {this};
  //dSpaceCollide(odeworld.space_id, &d, &nearCallbackCartpole);

  unsigned int begin_index = 0;
  double force = 0.f;
  
  force = bib::Utils::transform(motors[begin_index++], -1, 1, -MAX_TORQUE_SLIDER, MAX_TORQUE_SLIDER);
  dJointAddSliderForce(joints[0], force);
  
  Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
  dWorldStep(odeworld.world_id, WORLD_STEP);
  lock.release();

  dJointGroupEmpty(odeworld.contactgroup);

  update_state(current_step, max_step_per_instance);
}

void CartpoleWorld::update_state(uint current_step, uint max_step_per_instance) {
  uint begin_index = 0;
  
  internal_state[begin_index++] = dJointGetSliderPosition(joints[0]);
  internal_state[begin_index++] = dJointGetSliderPositionRate(joints[0]);
  internal_state[begin_index++] = dJointGetHingeAngle(joints[1]);
  internal_state[begin_index++] = dJointGetHingeAngleRate(joints[1]);
    
  if(add_time_in_state)
    internal_state[begin_index] = bib::Utils::transform(current_step, 0, max_step_per_instance, -1.f, 1.f);
  
  if(normalization){
    internal_state[0] = bib::Utils::transform(internal_state[0], -MAX_SLIDER_POSITON, MAX_SLIDER_POSITON, -1.f, 1.f);
    internal_state[1] = bib::Utils::transform(internal_state[1], -normalized_vector[0], normalized_vector[0], -1.f, 1.f);
    internal_state[2] = bib::Utils::transform(internal_state[2], -MAX_HINGE_ANGLE, MAX_HINGE_ANGLE, -1.f, 1.f);
    internal_state[3] = bib::Utils::transform(internal_state[3], -normalized_vector[1], normalized_vector[1], -1.f, 1.f);
  }
}

const std::vector<double>& CartpoleWorld::state() const {
  return internal_state;
}

unsigned int CartpoleWorld::activated_motors() const {
  return 1;
}

bool CartpoleWorld::final_state() const {
  if( fabs(dJointGetSliderPosition(joints[0])) >= MAX_SLIDER_POSITON )
    return true;
  
  if( fabs(dJointGetHingeAngle(joints[1])) >= MAX_HINGE_ANGLE)
    return true;
  
// continue even in goal state
//  if(goal_state())
//    return true;
  
  return false;
}

bool CartpoleWorld::goal_state() const {
  return fabs(dJointGetSliderPosition(joints[0])) <= 0.05 && fabs(dJointGetHingeAngle(joints[1])) <= M_PI/60.f;
}

void CartpoleWorld::resetPositions(std::vector<double>& stochasticity, const std::vector<double>& given_stoch) {
  dJointAddSliderForce(joints[0], 0);

  dMatrix3 R;
  dRFromEulerAngles(R, 0, 0, 0);

  for (ODEObject * o : bones) {
    dBodySetRotation(o->getID(), R);
    dBodySetForce(o->getID(), 0, 0, 0);
    dBodySetLinearVel(o->getID(), 0, 0, 0);
    dBodySetAngularVel(o->getID(), 0, 0, 0);
  }
  
  double theta, x;
  
  do {
    dBodySetPosition(bones[0]->getID(), 0, 0, CART_LARGER/2.f);
    dBodySetPosition(bones[1]->getID(), 0, 0, POLE_LENGTH/2.f+CART_LARGER/2.f);
    
    if(given_stoch.size() == 0)
      theta = bib::Utils::transform(bib::Utils::rand01(), 0.f, 1.f, -M_PI/18.f, M_PI/18.f);
    else
      theta = given_stoch[0];
//     theta=0;//determinist
    dRFromEulerAngles(R, 0, theta, 0);
    dBodySetRotation(bones[1]->getID(), R);
    
    double shifting = std::sin(theta)*POLE_LENGTH/2.f;
    
    if(given_stoch.size() == 0)
      x = bib::Utils::transform(bib::Utils::rand01(), 0, 1.f, shifting > 0 ? -0.5f : -0.5f - shifting, shifting < 0 ? 0.5f : 0.5f - shifting );
    else
      x = given_stoch[1];
//     x=0.2;//determinist
    
    dBodySetPosition(bones[0]->getID(), x + shifting, 0, CART_LARGER/2.f);
    dBodySetPosition(bones[1]->getID(), x, 0, POLE_LENGTH/2.f+CART_LARGER/2.f);
  } while(goal_state()); 
  
  dJointAddSliderForce(joints[0], 0);

  dJointGroupEmpty(odeworld.contactgroup);

  update_state(0, 1);
  stochasticity.resize(2);
  stochasticity[0]= theta;
  stochasticity[1]= x;
}
