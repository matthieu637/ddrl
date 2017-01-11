#include "HalfCheetahWorld.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include "ode/ode.h"

#include "bib/Utils.hpp"
#include "ODEFactory.hpp"


double softstepf(double x) {
  if (x<=0)
    return 0;
  else if(x <= 0.5)
    return 2*x*x;
  else if(x <= 1)
    return 1 - 2*(x-1)*(x-1);
  else
    return 1;
}

HalfCheetahWorld::HalfCheetahWorld(hcheetah_physics _phy) : odeworld(ODEFactory::getInstance()->createWorld()),
  phy(_phy) {

  dWorldSetGravity(odeworld.world_id, 0, 0.0, GRAVITY);

//   dContact contact[2];          // up to 3 contacts
  contact[0].surface.mode = dContactApprox0;
  contact[1].surface.mode = dContactApprox0;

  if (phy.approx == 1) {
    contact[0].surface.mode = contact[0].surface.mode | dContactApprox1;
    contact[1].surface.mode = contact[1].surface.mode | dContactApprox1;
  } else if (phy.approx == 2) {
    contact[0].surface.mode = contact[0].surface.mode | dContactApprox1_1;
    contact[1].surface.mode = contact[1].surface.mode | dContactApprox1_1;
  } else if (phy.approx == 3) {
    contact[0].surface.mode = contact[0].surface.mode | dContactApprox1_N;
    contact[1].surface.mode = contact[1].surface.mode | dContactApprox1_N;
  }

  if (phy.mu2 >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactMu2;
    contact[1].surface.mode = contact[1].surface.mode | dContactMu2;
  }

  if (phy.soft_cfm >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactSoftCFM;
    contact[1].surface.mode = contact[1].surface.mode | dContactSoftCFM;
  }

  if (phy.slip1 >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactSlip1;
    contact[1].surface.mode = contact[1].surface.mode | dContactSlip1;
  }

  if (phy.slip2 >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactSlip2;
    contact[1].surface.mode = contact[1].surface.mode | dContactSlip2;
  }

  if (phy.soft_erp >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactSoftERP;
    contact[1].surface.mode = contact[1].surface.mode | dContactSoftERP;
  }

  if (phy.bounce >= 0.0000f) {
    contact[0].surface.mode = contact[0].surface.mode | dContactBounce;
    contact[1].surface.mode = contact[1].surface.mode | dContactBounce;
  }

  contact[0].surface.mu = phy.mu;
  contact[0].surface.mu2 = phy.mu2;
  contact[0].surface.soft_cfm = phy.soft_cfm;
  contact[0].surface.slip1 = phy.slip1;
  contact[0].surface.slip2 = phy.slip2;
  contact[0].surface.soft_erp = phy.soft_erp;
  contact[0].surface.bounce = phy.bounce;

  contact[1].surface.mu = phy.mu;
  contact[1].surface.mu2 = phy.mu2;
  contact[1].surface.soft_cfm = phy.soft_cfm;
  contact[1].surface.slip1 = phy.slip1;
  contact[1].surface.slip2 = phy.slip2;
  contact[1].surface.soft_erp = phy.soft_erp;
  contact[1].surface.bounce = phy.bounce;

  createWorld();

  head_touch = false;
  fknee_touch = false;
  bknee_touch = false;
  penalty = 0;

  if(phy.predev == 0)
    internal_state.resize(18);
  else if(phy.predev == 1 || phy.predev == 10)
    internal_state.resize(18 - 4);
  else if(phy.predev == 2 || phy.predev == 11)
    internal_state.resize(18);
  else if(phy.predev == 3 || phy.predev == 12)
    internal_state.resize(18);

  update_state();
}

HalfCheetahWorld::~HalfCheetahWorld() {
  for (ODEObject * o : bones) {
    dGeomDestroy(o->getGeom());
    if(o->getID() != nullptr)
      dBodyDestroy(o->getID());
    delete o;
  }

  dGeomDestroy(ground);

  ODEFactory::getInstance()->destroyWorld(odeworld);
}

void HalfCheetahWorld::apply_armature(dMass* m, double k) {
  if(!phy.apply_armature)
    return;

  m->I[0] = m->I[0] + k;
  m->I[3] = m->I[3] + k;
  m->I[6] = m->I[6] + k;
}

void HalfCheetahWorld::apply_damping(dBodyID body, double v) {
  if(phy.damping == 1)
    dBodySetLinearDampingThreshold(body, v);
  else if(phy.damping == 2)
    dBodySetAngularDampingThreshold(body, v);
  else if(phy.damping == 3)
    dBodySetLinearDamping(body, v);
  else if(phy.damping == 4)
    dBodySetAngularDamping(body, v);
}

void HalfCheetahWorld::createWorld() {
  ground = ODEFactory::getInstance()->createGround(odeworld);

//   dWorldSetCFM(odeworld.world_id, 1.);
//   dWorldSetERP(odeworld.world_id, 1.);

//   <compiler inertiafromgeom='true' coordinate='local' angle='radian' settotalmass='14' />
  double density = 593.578f;//so total mass is 14

  //   <joint limited='true' damping='.01' armature='.1' stiffness='8' solreflimit='.02 1' solimplimit='0 .8 .03' />
//   dWorldSetLinearDamping(odeworld.world_id, .01);
  if(phy.damping == 1)
    dWorldSetLinearDampingThreshold(odeworld.world_id, .01);
  else if(phy.damping == 2)
    dWorldSetAngularDampingThreshold(odeworld.world_id, .01);
  else if(phy.damping == 3)
    dWorldSetLinearDamping(odeworld.world_id, .01);
  else if(phy.damping == 4)
    dWorldSetAngularDamping(odeworld.world_id, .01);
//   armature
//     Armature inertia (or rotor inertia) of all degrees of freedom created by this joint. These are constants added to the diagonal of the inertia matrix in generalized coordinates. They make the simulation more stable, and often increase physical realism. This is because when a motor is attached to the system with a transmission that amplifies the motor force by c, the inertia of the rotor (i.e. the moving part of the motor) is amplified by c*c. The same holds for gears in the early stages of planetary gear boxes. These extra inertias often dominate the inertias of the robot parts that are represented explicitly in the model, and the armature attribute is the way to model them.
//   stiffness
//   A positive value generates a spring force (linear in position) acting along the tendon. The equilibrium length of the spring corresponds to the tendon length when the model is in its initial configuration.
//   solreflimit, solimplimit
//   Constraint solver parameters for simulating tendon limits. See Solver parameters.

  //   <geom contype='1' conaffinity='0' condim='3' friction='.4 .1 .1' rgba='0.8 0.6 .4 1' solimp='0.0 0.8 0.01' solref='0.02 1' />
//   done in collision

  // <body name='torso' pos='0 0 .7'>
  dBodyID torso = dBodyCreate(odeworld.world_id);
  dBodySetPosition(torso, 0, 0, 0.7);

//   <joint name='rootx' type='slide' pos='0 0 0' axis='1 0 0' limited='false' damping='0' armature='0' stiffness='0' />
//   <joint name='rootz' type='slide' pos='0 0 0' axis='0 0 1' limited='false' damping='0' armature='0' stiffness='0' />
//   <joint name='rooty' type='hinge' pos='0 0 0' axis='0 1 0' limited='false' damping='0' armature='0' stiffness='0' />

  double radius = 0.046;
  double length_multiplier = 2.0f;

  //   <geom name='torso' type='capsule' fromto='-.5 0 0 .5 0 0' size='0.046' />
  dGeomID g_torso = dCreateCapsule(odeworld.space_id, radius, 1);
  dGeomSetBody(g_torso, torso);
  dMatrix3 R;
  dRFromAxisAndAngle(R, 0, 1, 0, 1.5708f);
  dGeomSetOffsetRotation(g_torso, R);

  //   <geom name='head' type='capsule' pos='.6 0 .1' axisangle='0 1 0 .87' size='0.046 .15' />
  dGeomID g_head = dCreateCapsule(odeworld.space_id, radius, 0.15*length_multiplier);
  dGeomSetBody(g_head, torso);
  dGeomSetOffsetPosition(g_head, 0.6, 0, 0.1);
  dMatrix3 R_head;
  dRFromAxisAndAngle(R_head, 0, 1, 0, 0.87);
  dGeomSetOffsetRotation(g_head, R_head);

  dMass m_torso;
  dMassSetCapsule(&m_torso, density, 3, radius, 1);
  dMass m_torso2;
  dMassSetCapsule(&m_torso2, density, 3, radius, 0.15*length_multiplier);
  dMassAdd(&m_torso, &m_torso2);
  apply_armature(&m_torso, 0.1);
  dBodySetMass(torso, &m_torso);

  //   <body name='bthigh' pos='-.5 0 0'>
  dBodyID bthigh = dBodyCreate(odeworld.world_id);
  dBodySetPosition(bthigh, -.5, 0., 0+0.7);

  //   <joint name='bthigh' type='hinge' pos='0 0 0' axis='0 1 0' range='-.52 1.05' stiffness='240' damping='6' />
  dJointID j_bthigh = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_bthigh, bthigh, torso);
  dJointSetHingeAxis(j_bthigh, 0, 1, 0);//no body attached no effect
  dJointSetHingeParam(j_bthigh, dParamLoStop, -0.52);
  dJointSetHingeParam(j_bthigh, dParamHiStop, 1.05);
  apply_damping(bthigh, 6);


  //   <geom name='bthigh' type='capsule' pos='.1 0 -.13' axisangle='0 1 0 -3.8' size='0.046 .145' />
  dGeomID g_bthigh = dCreateCapsule(odeworld.space_id, radius, 0.145*length_multiplier);
  dGeomSetBody(g_bthigh, bthigh);
  dGeomSetOffsetPosition(g_bthigh, .1, 0, -.13);
  dMatrix3 R_bthigh;
  dRFromAxisAndAngle(R_bthigh, 0, 1, 0, -3.8);
  dGeomSetOffsetRotation(g_bthigh, R_bthigh);

  dMass m_bthigh;
  dMassSetCapsule(&m_bthigh, density, 3, radius, 0.145*length_multiplier);
  apply_armature(&m_bthigh, 0.1);
  dBodySetMass(bthigh, &m_bthigh);

  //   <body name='bshin' pos='.16 0 -.25'>
  dBodyID bshin = dBodyCreate(odeworld.world_id);
  dBodySetPosition(bshin, .16 -.5, 0., -.25 + 0.7);

  //   <joint name='bshin' type='hinge' pos='0 0 0' axis='0 1 0' range='-.785 .785' stiffness='180' damping='4.5' />
  dJointID j_bshin = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_bshin, bshin, bthigh);
  dJointSetHingeAxis(j_bshin, 0, 1, 0);
  dJointSetHingeParam(j_bshin, dParamLoStop, -.785);
  dJointSetHingeParam(j_bshin, dParamHiStop, .785);
  apply_damping(bshin, 4.5);

  //   <geom name='bshin' type='capsule' pos='-.14 0 -.07' axisangle='0 1 0 -2.03' size='0.046 .15' rgba='0.9 0.6 0.6 1' />
  dGeomID g_bshin = dCreateCapsule(odeworld.space_id, radius, .15*length_multiplier);
  dGeomSetBody(g_bshin, bshin);
  dGeomSetOffsetPosition(g_bshin, -.14, 0, -.07);
  dMatrix3 R_bshin;
  dRFromAxisAndAngle(R_bshin, 0, 1, 0, -2.03);
  dGeomSetOffsetRotation(g_bshin, R_bshin);

  dMass m_bshin;
  dMassSetCapsule(&m_bshin, density, 3, radius, .15*length_multiplier);
  apply_armature(&m_bshin, 0.1);
  dBodySetMass(bshin, &m_bshin);

  //   <body name='bfoot' pos='-.28 0 -.14'>
  dBodyID bfoot = dBodyCreate(odeworld.world_id);
  dBodySetPosition(bfoot, .16 -.5 -.28, 0., -.25 + 0.7 -.14);

  //   <joint name='bfoot' type='hinge' pos='0 0 0' axis='0 1 0' range='-.4 .785' stiffness='120' damping='3' />
  dJointID j_bfoot = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_bfoot, bfoot, bshin);
  dJointSetHingeAxis(j_bfoot, 0, 1, 0);
  dJointSetHingeParam(j_bfoot, dParamLoStop, -.4);
  dJointSetHingeParam(j_bfoot, dParamHiStop, .785);
  apply_damping(bfoot, 3.);

  //   <geom name='bfoot' type='capsule' pos='.03 0 -.097' axisangle='0 1 0 -.27' size='0.046 .094' rgba='0.9 0.6 0.6 1' />
  dGeomID g_bfoot = dCreateCapsule(odeworld.space_id, radius, .094*length_multiplier);
  dGeomSetBody(g_bfoot, bfoot);
  dGeomSetOffsetPosition(g_bfoot, .03, 0, -.097);
  dMatrix3 R_bfoot;
  dRFromAxisAndAngle(R_bfoot, 0, 1, 0, -.27);
  dGeomSetOffsetRotation(g_bfoot, R_bfoot);

  dMass m_bfoot;
  dMassSetCapsule(&m_bfoot, density, 3, radius, .094*length_multiplier);
  apply_armature(&m_bfoot, 0.1);
  dBodySetMass(bfoot, &m_bfoot);

  //   <body name='fthigh' pos='.5 0 0'>
  dBodyID fthigh = dBodyCreate(odeworld.world_id);
  dBodySetPosition(fthigh, .5, 0, 0.7);

  //   <joint name='fthigh' type='hinge' pos='0 0 0' axis='0 1 0' range='-1 .7' stiffness='180' damping='4.5' />
  dJointID j_fthigh = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_fthigh, fthigh, torso);
  dJointSetHingeAxis(j_fthigh, 0, 1, 0);
  dJointSetHingeParam(j_fthigh, dParamLoStop, -1);
  dJointSetHingeParam(j_fthigh, dParamHiStop, .7);
  apply_damping(fthigh, 4.5);

  //   <geom name='fthigh' type='capsule' pos='-.07 0 -.12' axisangle='0 1 0 .52' size='0.046 .133' />
  dGeomID g_fthigh = dCreateCapsule(odeworld.space_id, radius, .133*length_multiplier);
  dGeomSetBody(g_fthigh, fthigh);
  dGeomSetOffsetPosition(g_fthigh, -.07, 0, -.12);
  dMatrix3 R_fthigh;
  dRFromAxisAndAngle(R_fthigh, 0, 1, 0, .52);
  dGeomSetOffsetRotation(g_fthigh, R_fthigh);

  dMass m_fthigh;
  dMassSetCapsule(&m_fthigh, density, 3, radius, .133*length_multiplier);
  apply_armature(&m_fthigh, 0.1);
  dBodySetMass(fthigh, &m_fthigh);

  //   <body name='fshin' pos='-.14 0 -.24'>
  dBodyID fshin = dBodyCreate(odeworld.world_id);
  dBodySetPosition(fshin, .5 -.14, 0, 0.7 -.24);

  //   <joint name='fshin' type='hinge' pos='0 0 0' axis='0 1 0' range='-1.2 .87' stiffness='120' damping='3' />
  dJointID j_fshin = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_fshin, fshin, fthigh);
  dJointSetHingeAxis(j_fshin, 0, 1, 0);
  dJointSetHingeParam(j_fshin, dParamLoStop, -1.2);
  dJointSetHingeParam(j_fshin, dParamHiStop, .87);
  apply_damping(fshin, 3.);

  //   <geom name='fshin' type='capsule' pos='.065 0 -.09' axisangle='0 1 0 -.6' size='0.046 .106' rgba='0.9 0.6 0.6 1' />
  dGeomID g_fshin = dCreateCapsule(odeworld.space_id, radius, .106*length_multiplier);
  dGeomSetBody(g_fshin, fshin);
  dGeomSetOffsetPosition(g_fshin, .065, 0, -.09);
  dMatrix3 R_fshin;
  dRFromAxisAndAngle(R_fshin, 0, 1, 0, -.6);
  dGeomSetOffsetRotation(g_fshin, R_fshin);

  dMass m_fshin;
  dMassSetCapsule(&m_fshin, density, 3, radius, .106*length_multiplier);
  apply_armature(&m_fshin, 0.1);
  dBodySetMass(fshin, &m_fshin);

  //   <body name='ffoot' pos='.13 0 -.18'>
  dBodyID ffoot = dBodyCreate(odeworld.world_id);
  dBodySetPosition(ffoot, .5 -.14 + .13, 0, 0.7 -.24 -.18);

  //   <joint name='ffoot' type='hinge' pos='0 0 0' axis='0 1 0' range='-.5 .5' stiffness='60' damping='1.5' />
  dJointID j_ffoot = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_ffoot, ffoot, fshin);
  dJointSetHingeAxis(j_ffoot, 0, 1, 0);
  dJointSetHingeParam(j_ffoot, dParamLoStop, -.5);
  dJointSetHingeParam(j_ffoot, dParamHiStop, .5);
  apply_damping(ffoot, 1.5);

  //   <geom name='ffoot' type='capsule' pos='.045 0 -.07' axisangle='0 1 0 -.6' size='0.046 .07' rgba='0.9 0.6 0.6 1' />
  dGeomID g_ffoot = dCreateCapsule(odeworld.space_id, radius, .07*length_multiplier);
  dGeomSetBody(g_ffoot, ffoot);
  dGeomSetOffsetPosition(g_ffoot, .045, 0, -.07);
  dMatrix3 R_ffoot;
  dRFromAxisAndAngle(R_ffoot, 0, 1, 0, -.6);
  dGeomSetOffsetRotation(g_ffoot, R_ffoot);

  dMass m_ffoot;
  dMassSetCapsule(&m_ffoot, density, 3, radius, .07*length_multiplier);
  apply_armature(&m_ffoot, 0.1);
  dBodySetMass(ffoot, &m_ffoot);

  joints.push_back(j_bthigh);
  joints.push_back(j_bshin);
  joints.push_back(j_bfoot);
  joints.push_back(j_fthigh);
  joints.push_back(j_fshin);
  joints.push_back(j_ffoot);

  bones.push_back(new ODEObject(nullptr, m_torso2, g_head, 0,0,0.,density, m_torso2.mass));
  bones.push_back(new ODEObject(torso, m_torso, g_torso, 0,0,0.,density, m_torso.mass));
  bones.push_back(new ODEObject(bthigh, m_bthigh, g_bthigh, 0,0,0.,density, m_bthigh.mass));
  bones.push_back(new ODEObject(bshin, m_bshin, g_bshin, 0,0,0.,density, m_bshin.mass));
  bones.push_back(new ODEObject(bfoot, m_bfoot, g_bfoot, 0,0,0.,density, m_bfoot.mass));
  bones.push_back(new ODEObject(fthigh, m_fthigh, g_fthigh, 0,0,0.,density, m_fthigh.mass));
  bones.push_back(new ODEObject(fshin, m_fshin, g_fshin, 0,0,0.,density, m_fshin.mass));
  bones.push_back(new ODEObject(ffoot, m_ffoot, g_ffoot, 0,0,0.,density, m_ffoot.mass));

#ifndef NDEBUG
  double mass_sum = 0;
  for(auto a : bones) {
//     LOG_DEBUG(a->getMassValue());
    mass_sum += a->getMassValue();
  }
//   LOG_DEBUG("");
  ASSERT(mass_sum >= 14.f - 0.001f && mass_sum <= 14.f + 0.001f, "sum mass : " << mass_sum);
#endif

//   bib::Logger::PRINT_ELEMENTS(second_bone->getMass().I, 9, "pole inertia : ");
}

void nearCallbackHalfCheetah(void* data, dGeomID o1, dGeomID o2) {
  nearCallbackDataHalfCheetah* d = reinterpret_cast<nearCallbackDataHalfCheetah*>(data);
  HalfCheetahWorld* inst = d->inst;

  // only collide things with the ground
  if(o1 != inst->ground && o2 != inst->ground)
    return;

//   <geom contype='1' conaffinity='0' condim='3' friction='.4 .1 .1' rgba='0.8 0.6 .4 1' solimp='0.0 0.8 0.01' solref='0.02 1' />
  if(o1 == inst->bones[5]->getGeom() || o2 == inst->bones[5]->getGeom())
    inst->fknee_touch = true;
  if(o1 == inst->bones[2]->getGeom() || o2 == inst->bones[2]->getGeom())
    inst->bknee_touch = true;
  if(o1 == inst->bones[0]->getGeom() || o2 == inst->bones[0]->getGeom())
    inst->head_touch = true;

  if (int numc = dCollide (o1,o2,2,&inst->contact[0].geom,sizeof(dContact))) {
    for (int i=0; i<numc; i++) {
      dJointID c = dJointCreateContact (inst->odeworld.world_id,inst->odeworld.contactgroup,&inst->contact[i]);
      dJointAttach (c, dGeomGetBody(o1), dGeomGetBody(o2));
    }
//     LOG_DEBUG(numc);
  }

}

void HalfCheetahWorld::step(const vector<double>& _motors) {
  std::vector<double> motors(_motors);
  if(phy.predev == 1 || phy.predev == 2 || phy.predev == 3) {
    motors.resize(6);
    motors[2] = 0;
    motors[3] = _motors[2];
    motors[4] = _motors[3];
    motors[5] = 0;
  } else if (phy.predev == 10 || phy.predev == 11 || phy.predev == 12) {
    motors.resize(6);
    motors[1] = 0;
    motors[2] = _motors[1];
    motors[3] = _motors[2];
    motors[4] = 0;
    motors[5] = _motors[3];
  }
  
  if(phy.from_predev == 1 || phy.from_predev == 2 || phy.from_predev == 3){
    motors[2] = _motors[4];
    motors[3] = _motors[2];
    motors[4] = _motors[3];
//     motors[5] = _motors[5];
  } else if(phy.from_predev == 10 || phy.from_predev == 11 || phy.from_predev == 12){
    motors[1] = _motors[4];
    motors[2] = _motors[1];
    motors[3] = _motors[2];
    motors[4] = _motors[5];
    motors[5] = _motors[3];
  }

  head_touch = false;
  fknee_touch = false;
  bknee_touch = false;
  penalty = 0;

  nearCallbackDataHalfCheetah d = {this};
  dSpaceCollide(odeworld.space_id, &d, &nearCallbackHalfCheetah);

  unsigned int begin_index = 0;

  if(phy.control == 1) {
    begin_index = 0;
    double f_bthigh = bib::Utils::transform(motors[begin_index++], -1, 1, -120, 120);
    double f_bshin = bib::Utils::transform(motors[begin_index++], -1, 1, -90, 90);
    double f_bfoot = bib::Utils::transform(motors[begin_index++], -1, 1, -60, 60);
    double f_fthigh = bib::Utils::transform(motors[begin_index++], -1, 1, -120, 120);
    double f_fshin = bib::Utils::transform(motors[begin_index++], -1, 1, -60, 60);
    double f_ffoot = bib::Utils::transform(motors[begin_index++], -1, 1, -30, 30);

    begin_index = 0;
    dJointAddHingeTorque(joints[begin_index++], f_bthigh);
    dJointAddHingeTorque(joints[begin_index++], f_bshin);
    dJointAddHingeTorque(joints[begin_index++], f_bfoot);
    dJointAddHingeTorque(joints[begin_index++], f_fthigh);
    dJointAddHingeTorque(joints[begin_index++], f_fshin);
    dJointAddHingeTorque(joints[begin_index++], f_ffoot);
  } else if(phy.control==2 || phy.control==3) { //origin paper
    double p_bthigh = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[0])) - 0.05 * dJointGetHingeAngleRate(
                                           joints[0]));
    double p_bshin = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[1])) - 0.05 * dJointGetHingeAngleRate(joints[1]));
    double p_bfoot = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[2])) - 0.05 * dJointGetHingeAngleRate(joints[2]));
    double p_ffthigh = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[3])) - 0.05 * dJointGetHingeAngleRate(
                                            joints[3]));
    double p_fshin = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[4])) - 0.05 * dJointGetHingeAngleRate(joints[4]));
    double p_ffoot = (2.0f/M_PI) * atan(-2.0f*(dJointGetHingeAngle(joints[5])) - 0.05 * dJointGetHingeAngleRate(joints[5]));

    begin_index = 0;
    double a_bthigh = motors[begin_index++];
    double a_bshin = motors[begin_index++];
    double a_bfoot = motors[begin_index++];
    double a_fthigh = motors[begin_index++];
    double a_fshin = motors[begin_index++];
    double a_ffoot = motors[begin_index++];
    if(phy.control==3) {
      a_bthigh = bib::Utils::transform(a_bthigh, -1.f, 1.f, -2.f, 2.f);
      a_bshin = bib::Utils::transform(a_bshin, -1.f, 1.f, -2.f, 2.f);
      a_bfoot = bib::Utils::transform(a_bfoot, -1.f, 1.f, -2.f, 2.f);
      a_fthigh = bib::Utils::transform(a_fthigh, -1.f, 1.f, -2.f, 2.f);
      a_fshin = bib::Utils::transform(a_fshin, -1.f, 1.f, -2.f, 2.f);
      a_ffoot = bib::Utils::transform(a_ffoot, -1.f, 1.f, -2.f, 2.f);
    }

    double f_bthigh = 120* std::min(std::max((double)-1., p_bthigh+a_bthigh), (double)1.);
    double f_bshin = 90*std::min(std::max((double)-1., p_bshin+a_bshin), (double)1.);
    double f_bfoot = 60*std::min(std::max((double)-1., p_bfoot+a_bfoot), (double)1.);
    double f_fthigh = 90*std::min(std::max((double)-1., p_ffthigh+a_fthigh), (double)1.);
    double f_fshin = 60*std::min(std::max((double)-1., p_fshin+a_fshin), (double)1.);
    double f_ffoot = 30*std::min(std::max((double)-1., p_ffoot+a_ffoot), (double)1.);

    begin_index = 0;
    dJointAddHingeTorque(joints[begin_index++], f_bthigh);
    dJointAddHingeTorque(joints[begin_index++], f_bshin);
    dJointAddHingeTorque(joints[begin_index++], f_bfoot);
    dJointAddHingeTorque(joints[begin_index++], f_fthigh);
    dJointAddHingeTorque(joints[begin_index++], f_fshin);
    dJointAddHingeTorque(joints[begin_index++], f_ffoot);

    if(phy.reward == 2) {
      begin_index = 0;
      double sub_pelnalty = 0.f;
      sub_pelnalty += std::max(fabs(p_bthigh+motors[begin_index++]) - 1.f, (double) 0.f);
      sub_pelnalty += std::max(fabs(p_bshin+motors[begin_index++]) - 1.f, (double) 0.f);
      sub_pelnalty += std::max(fabs(p_bfoot+motors[begin_index++]) - 1.f, (double) 0.f);
      sub_pelnalty += std::max(fabs(p_ffthigh+motors[begin_index++]) - 1.f, (double) 0.f);
      sub_pelnalty += std::max(fabs(p_fshin+motors[begin_index++]) - 1.f, (double) 0.f);
      sub_pelnalty += std::max(fabs(p_ffoot+motors[begin_index++]) - 1.f, (double) 0.f);
      penalty += -0.05 * sub_pelnalty;

//       already managed by stops
//       sub_pelnalty = 0.f;
//       sub_pelnalty += std::min(fabs(p_bthigh*120) , (double) 50.f);
//       sub_pelnalty += std::min(fabs(p_bshin*90) , (double) 50.f);
//       sub_pelnalty += std::min(fabs(p_bfoot*60) , (double) 50.f);
//       sub_pelnalty += std::min(fabs(p_ffthigh*90) , (double) 50.f);
//       sub_pelnalty += std::min(fabs(p_fshin*60) , (double) 50.f);
//       sub_pelnalty += std::min(fabs(p_ffoot*30) , (double) 50.f);
//       penalty += -0.1 * sub_pelnalty;
    }
  }

  if(phy.reward == 1 || phy.reward == 3) {
    for (auto a : motors)
      penalty += a*a;
    penalty = - 1e-1 * 0.5 * penalty;
  }

  Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
  dWorldStep(odeworld.world_id, WORLD_STEP);
  lock.release();

  dJointGroupEmpty(odeworld.contactgroup);

  update_state();
}

void HalfCheetahWorld::update_state() {
  uint begin_index = 0;

  dBodyID torso = bones[1]->getID();
  const dReal* root_pos = dBodyGetPosition(torso);
  const dReal* root_vel = dBodyGetLinearVel(torso);
  const dReal* root_angvel = dBodyGetAngularVel(torso);
  const dReal* root_rot = dBodyGetQuaternion(torso);
  ASSERT(root_rot[3] <= 1 , "not normalized");
  double s = dSqrt(1.0f-root_rot[3]*root_rot[3]);

  std::list<double> substate;

  substate.push_back(root_pos[0]);//- rootx     slider      position (m)
  substate.push_back(root_pos[2]);//- rootz     slider      position (m)
  substate.push_back(s <= 0.0000001f ? root_rot[2] : root_rot[2]/s) ;// - rooty     hinge       angle (rad)
  substate.push_back(dJointGetHingeAngle(joints[0])); //- bthigh    hinge       angle (rad)
  substate.push_back(dJointGetHingeAngle(joints[1]));
  substate.push_back(dJointGetHingeAngle(joints[2]));
  substate.push_back(dJointGetHingeAngle(joints[3]));
  substate.push_back(dJointGetHingeAngle(joints[4]));
  substate.push_back(dJointGetHingeAngle(joints[5]));

  substate.push_back(root_vel[0]);//- rootx     slider      velocity (m/s)
  substate.push_back(root_vel[2]);//- rootz     slider      velocity (m/s)
  substate.push_back(root_angvel[1]);//- rooty     hinge       angular velocity (rad/s)
  substate.push_back(dJointGetHingeAngleRate(joints[0])); //- bthigh    hinge       angular velocity (rad/s)
  substate.push_back(dJointGetHingeAngleRate(joints[1]));
  substate.push_back(dJointGetHingeAngleRate(joints[2]));
  substate.push_back(dJointGetHingeAngleRate(joints[3]));
  substate.push_back(dJointGetHingeAngleRate(joints[4]));
  substate.push_back(dJointGetHingeAngleRate(joints[5]));

  ASSERT(substate.size() == 18, "wrong indices");

  if((phy.from_predev == 0 && (phy.predev == 0 || phy.predev == 2 || phy.predev == 11 || phy.predev == 3 || phy.predev == 12)) ||
    phy.from_predev == 2 || phy.from_predev == 11 || phy.from_predev == 3 || phy.from_predev == 12) {
    std::copy(substate.begin(), substate.end(), internal_state.begin());
  
    if(phy.predev == 3){
      internal_state[17] = 0.0f;
      internal_state[14] = 0.0f;
      internal_state[8]  = 0.0f;
      internal_state[5]  = 0.0f;
    } else if(phy.predev == 12){
      internal_state[16] = 0.0f;
      internal_state[13] = 0.0f;
      internal_state[7]  = 0.0f;
      internal_state[4]  = 0.0f;
    }
  } else {
    std::list<uint> later;

    if(phy.predev == 1 || phy.from_predev == 1) {
      auto it = substate.begin();
      std::advance(it, 17);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 14);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 8);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 5);
      substate.erase(it);

      std::copy(substate.begin(), substate.end(), internal_state.begin());
    } else if(phy.predev == 10 || phy.from_predev == 10) {
      auto it = substate.begin();
      std::advance(it, 16);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 13);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 7);
      later.push_back(*it);
      substate.erase(it);

      it = substate.begin();
      std::advance(it, 4);
      later.push_back(*it);
      substate.erase(it);
      std::copy(substate.begin(), substate.end(), internal_state.begin());
    }
    
    if(phy.from_predev != 0)
      std::copy(later.begin(), later.end(), internal_state.begin() + substate.size());
  }

//   bib::Logger::PRINT_ELEMENTS(internal_state);

//   if(fknee_touch){
//     LOG_DEBUG("front touched");
//   }
//   if(bknee_touch){
//     LOG_DEBUG("back touched");
//   }

  if(phy.reward == 2 || phy.reward == 3) {
    if(head_touch)
      penalty += -1;
    if(fknee_touch)
      penalty += -1;
    if(bknee_touch)
      penalty += -1;

    reward = penalty + root_vel[0];
  } else if(phy.reward == 1) {
    reward = penalty + root_vel[0];
  }
}

const std::vector<double>& HalfCheetahWorld::state() const {
  return internal_state;
}

unsigned int HalfCheetahWorld::activated_motors() const {
  if(phy.predev != 0)
    return 4;
  return 6;
}

bool HalfCheetahWorld::final_state() const {
  return head_touch;
}

double HalfCheetahWorld::performance() const {
  if(final_state())
    return -1000;
  return reward;
}

void HalfCheetahWorld::resetPositions(std::vector<double>&, const std::vector<double>&) {
  for (ODEObject * o : bones) {
    dGeomDestroy(o->getGeom());
    if(o->getID() != nullptr)
      dBodyDestroy(o->getID());
    delete o;
  }
  bones.clear();

  for(auto j : joints)
    dJointDestroy(j);
  joints.clear();

  dGeomDestroy(ground);

//   ODEFactory::getInstance()->destroyWorld(odeworld);

  createWorld();
  
  head_touch = false;
  fknee_touch = false;
  bknee_touch = false;
  penalty = 0;
  
  update_state();
}
