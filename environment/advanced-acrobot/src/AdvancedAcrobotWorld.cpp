#include <bib/Utils.hpp>
#include "AdvancedAcrobotWorld.hpp"
#include "ODEFactory.hpp"
#include <ode/ode.h>
#include <functional>

AdvancedAcrobotWorld::AdvancedAcrobotWorld(
  const std::vector<bone_joint> &_types, const std::vector<bool> &_actuators)
  : odeworld(ODEFactory::getInstance()->createWorld()),
    types(_types),
    actuators(_actuators) {
  ASSERT(_types.size() == (_actuators.size() - 1),
         "actuators " << _actuators.size() << " not compatible with types "
         << _types.size());
  dWorldSetGravity(odeworld.world_id, 0, 0.0, GRAVITY);

  createWorld(_types);

  internal_state.push_back(dJointGetHingeAngle(joints[0]));
  internal_state.push_back(dJointGetHingeAngleRate(joints[0]));
  for (unsigned int i = 0; i < _types.size(); i++)
    if (_types[i] == HINGE) {
      internal_state.push_back(dJointGetHingeAngle(joints[i + 1]));
      internal_state.push_back(dJointGetHingeAngleRate(joints[i + 1]));
    } else {
      internal_state.push_back(dJointGetSliderPosition(joints[i + 1]));
      internal_state.push_back(dJointGetSliderPositionRate(joints[i + 1]));
    }

  _activated_motors = 0;
  for (bool b : _actuators)
    if (b) _activated_motors++;
}

AdvancedAcrobotWorld::~AdvancedAcrobotWorld() {
  for (ODEObject *o : bones) {
    dGeomDestroy(o->getGeom());
    dBodyDestroy(o->getID());
    delete o;
  }

  dGeomDestroy(ground);

  ODEFactory::getInstance()->destroyWorld(odeworld);
}

void AdvancedAcrobotWorld::createWorld(const std::vector<bone_joint> &types) {
  ASSERT(types.size() > 0, "number of types :" << types.size());

  float bone_length = BONE_LENGTH;
  float bone_larger = BONE_LARGER;
  float starting_z = STARTING_Z + bone_length * types.size();

  ground = ODEFactory::getInstance()->createGround(odeworld);

  //  Create the first fixed bone with it's joint
  ODEObject *first_bone = ODEFactory::getInstance()->createBox(
                            odeworld, 0., 0, starting_z, bone_larger, bone_larger, bone_length,
                            BONE_DENSITY, BONE_MASS, true);
  bones.push_back(first_bone);

  dJointID first_hinge = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(first_hinge, 0, first_bone->getID());
  dJointSetHingeAnchor(first_hinge, 0, 0, starting_z + bone_length / 2.);
  dJointSetHingeAxis(first_hinge, 0, 1, 0);
  joints.push_back(first_hinge);

  //  Create the other bone relative to the first one
  for (bone_joint type : types) {
    float my_starting_z = starting_z - bone_length * bones.size();
    ODEObject *next = ODEFactory::getInstance()->createBox(
                        odeworld, 0., 0, my_starting_z, bone_larger, bone_larger, bone_length,
                        BONE_DENSITY, BONE_MASS, true);

    if (type == HINGE) {
      dJointID next_hinge = dJointCreateHinge(odeworld.world_id, nullptr);
      dJointAttach(next_hinge, bones.back()->getID(), next->getID());
      dJointSetHingeAnchor(next_hinge, 0, 0, my_starting_z + bone_length / 2.);
      dJointSetHingeAxis(next_hinge, 0, 1, 0);
      joints.push_back(next_hinge);
    } else {  // type == SLIDER
      dJointID next_slider = dJointCreateSlider(odeworld.world_id, nullptr);
      dJointAttach(next_slider, bones.back()->getID(), next->getID());
      dJointSetSliderAxis(next_slider, 0, 0, 1);
      dJointSetSliderParam(next_slider, dParamLoStop, -bone_length);
      dJointSetSliderParam(next_slider, dParamHiStop, 0.);
      joints.push_back(next_slider);
    }

    bones.push_back(next);
  }
}

void nearCallback(void *data, dGeomID o1, dGeomID o2) {
  int n;
  nearCallbackData *d = (nearCallbackData *)data;
  AdvancedAcrobotWorld *inst = d->inst;

  // exit without doing anything if the two bodies are connected by a joint
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnectedExcluding(b1, b2, dJointTypeContact)) return;

  const int N = 10;
  dContact contact[N];
  n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
  if (n > 0) {
    for (int i = 0; i < n; i++) {
      contact[i].surface.mode = dContactSlip1 | dContactSlip2 |
                                dContactSoftERP | dContactSoftCFM |
                                dContactApprox1;
      contact[i].surface.mu = dInfinity;
      contact[i].surface.slip1 = 0.5;
      contact[i].surface.slip2 = 0.5;
      contact[i].surface.soft_erp = 0.95;
      contact[i].surface.soft_cfm = 0.5;

      dJointID c = dJointCreateContact(
                     inst->odeworld.world_id, inst->odeworld.contactgroup, &contact[i]);
      dJointAttach(c, dGeomGetBody(contact[i].geom.g1),
                   dGeomGetBody(contact[i].geom.g2));
    }
  }
}

void AdvancedAcrobotWorld::step(const vector<float> &motors) {
  nearCallbackData d = {this};

  dSpaceCollide(odeworld.space_id, &d, &nearCallback);

  unsigned int begin_index = 0;
  if (actuators[0])
    dJointAddHingeTorque(
      joints[0], bib::Utils::transform(motors[begin_index++], -1, 1,
                                       -MAX_TORQUE_HINGE, MAX_TORQUE_HINGE));

  for (unsigned int i = 1; i < actuators.size(); i++)
    if (actuators[i]) {
      if (types[i - 1] == HINGE)
        dJointAddHingeTorque(
          joints[i],
          bib::Utils::transform(motors[begin_index++], -1, 1,
                                -MAX_TORQUE_HINGE, MAX_TORQUE_HINGE));
      else
        dJointAddSliderForce(
          joints[i],
          bib::Utils::transform(motors[begin_index++], -1, 1,
                                -MAX_TORQUE_SLIDER, MAX_TORQUE_SLIDER));
    }

  ASSERT(begin_index == motors.size(),
         "wrong number of motors " << begin_index << " " << motors.size() << " "
         << actuators.size());

  Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
  dWorldStep(odeworld.world_id, WORLD_STEP);
  lock.release();

  dJointGroupEmpty(odeworld.contactgroup);

  begin_index = 0;
  internal_state[begin_index++] = dJointGetHingeAngle(joints[0]);
  internal_state[begin_index++] = dJointGetHingeAngleRate(joints[0]);
  for (unsigned int i = 0; i < types.size(); i++)
    if (types[i] == HINGE) {
      internal_state[begin_index++] = dJointGetHingeAngle(joints[i + 1]);
      internal_state[begin_index++] = dJointGetHingeAngleRate(joints[i + 1]);
    } else {
      internal_state[begin_index++] = dJointGetSliderPosition(joints[i + 1]);
      internal_state[begin_index++] =
        dJointGetSliderPositionRate(joints[i + 1]);
    }
}

const std::vector<float> &AdvancedAcrobotWorld::state() const {
  return internal_state;
}

unsigned int AdvancedAcrobotWorld::activated_motors() {
  return _activated_motors;
}

std::vector<float> *AdvancedAcrobotWorld::current_joint_forces() const {
  std::vector<float> *forces = new std::vector<float>;

  //     internal_state.push_back(dJointGetHingeAngle(joints[0]));
  //     internal_state.push_back(dJointGetHingeAngleRate(joints[0]));
  //
  //     for(unsigned int i =0; i<types.size(); i++)
  //         if(types[i] == HINGE) {
  //             internal_state.push_back(dJointGetHingeAngle(joints[i+1]));
  //             internal_state.push_back(dJointGetHingeAngleRate(joints[i+1]));
  //         }
  //         else {
  //             internal_state.push_back(dJointGetSliderPosition(joints[i]));
  //             internal_state.push_back(dJointGetSliderPositionRate(joints[i]));
  //         }

  return forces;
}

void AdvancedAcrobotWorld::resetPositions() {
  float starting_z = STARTING_Z + BONE_LENGTH * types.size();

  if (actuators[0]) dJointAddHingeTorque(joints[0], 0);

  for (unsigned int i = 1; i < actuators.size(); i++)
    if (actuators[i]) {
      if (types[i - 1] == HINGE)
        dJointAddHingeTorque(joints[i], 0);
      else
        dJointAddSliderForce(joints[i], 0);
    }

  dMatrix3 R;
  dRFromEulerAngles(R, 0, 0, 0);

  unsigned int i = 0;
  for (ODEObject *o : bones) {
    dBodySetRotation(o->getID(), R);
    dBodySetForce(o->getID(), 0, 0, 0);
    dBodySetLinearVel(o->getID(), 0, 0, 0);
    dBodySetAngularVel(o->getID(), 0, 0, 0);
    dBodySetPosition(o->getID(), 0, 0, starting_z - BONE_LENGTH * i++);
  }

  Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
  dWorldStep(odeworld.world_id, WORLD_STEP);
  lock.release();

  dJointGroupEmpty(odeworld.contactgroup);
}

float AdvancedAcrobotWorld::perf() const {
  float normalize = 2 * BONE_LENGTH * types.size();

  dVector3 result;
  dBodyGetRelPointPos(bones.back()->getID(), 0, 0, 0, result);

  return (result[2] - STARTING_Z) / normalize;
}
