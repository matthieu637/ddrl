#include "HumanoidWorld.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include "ode/ode.h"

#include "bib/Utils.hpp"
#include "ODEFactory.hpp"

// #define DEBUG_MOTOR_BY_MOTOR

HumanoidWorld::HumanoidWorld(humanoid_physics _phy) : odeworld(ODEFactory::getInstance()->createWorld(false)),
  phy(_phy), body_mass(11), qfrc_actuator(17, 0.f), gears(17) {

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
  
  //   <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  const double gear_abdomen_y = 100 * 0.4f;
  const double gear_abdomen_z = 100 * 0.4f;
  const double gear_abdomen_x = 100 * 0.4f;
  const double gear_right_hip_x = 100 * 0.4f;
  const double gear_right_hip_z = 100 * 0.4f;
  const double gear_right_hip_y = 300 * 0.4f;
  const double gear_right_knee = 200 * 0.4f;
  const double gear_left_hip_x = 100 * 0.4f;
  const double gear_left_hip_z = 100 * 0.4f;
  const double gear_left_hip_y = 300 * 0.4f;
  const double gear_left_knee = 200 * 0.4f;
  const double gear_right_shoulder1 = 25 * 0.4f;
  const double gear_right_shoulder2 = 25 * 0.4f;
  const double gear_right_elbow = 25 * 0.4f;
  const double gear_left_shoulder1 = 25 * 0.4f;
  const double gear_left_shoulder2 = 25 * 0.4f;
  const double gear_left_elbow = 25 * 0.4f;
  
  unsigned int begin_index = 0;
  gears[begin_index++] = gear_abdomen_z;
  gears[begin_index++] = gear_abdomen_y;
  gears[begin_index++] = gear_abdomen_x;
  //amotor order changes
  gears[begin_index++] = gear_right_hip_y;
  gears[begin_index++] = gear_right_hip_z;
  gears[begin_index++] = gear_right_hip_x;
  //--
  gears[begin_index++] = gear_right_knee;
  //amotor order changes
  gears[begin_index++] = gear_left_hip_y;
  gears[begin_index++] = gear_left_hip_z;
  gears[begin_index++] = gear_left_hip_x;
  //--
  gears[begin_index++] = gear_left_knee;
  gears[begin_index++] = gear_right_shoulder1;
  gears[begin_index++] = gear_right_shoulder2;
  gears[begin_index++] = gear_right_elbow;
  gears[begin_index++] = gear_left_shoulder1;
  gears[begin_index++] = gear_left_shoulder2;
  gears[begin_index++] = gear_left_elbow;

  createWorld();
  
  mass_sum = 0.f;
  
  begin_index = 0;
  for(auto a : bones) {
    if(a->getID() != nullptr){
      body_mass[begin_index++] = a->getMassValue();
      mass_sum += a->getMassValue();
    }
  }
  
#ifndef NDEBUG
  LOG_DEBUG("total mass : " << mass_sum);
  ASSERT(mass_sum >= 39.645f - 0.001f && mass_sum <= 39.645f + 0.001f, "sum mass : " << mass_sum);
#endif

  reward = ALIVE_BONUS;
  pos_before = 0.f;

  if(!phy.additional_sensors)
    internal_state.resize(22+23);
  else
    internal_state.resize(22+23+110+60+17);
  
  std::fill(internal_state.begin(), internal_state.end(), 0.f);

  update_state();
}

HumanoidWorld::~HumanoidWorld() {
  for (ODEObject * o : bones) {
    dGeomDestroy(o->getGeom());
    if(o->getID() != nullptr)
      dBodyDestroy(o->getID());
    delete o;
  }

  dGeomDestroy(ground);

  ODEFactory::getInstance()->destroyWorld(odeworld);
}

void HumanoidWorld::apply_armature(dMass* m, double k) {
  if(!phy.apply_armature)
    return;

  m->I[0] = m->I[0] + k;
  m->I[3] = m->I[3] + k;
  m->I[6] = m->I[6] + k;
}

// void HumanoidWorld::copy_inertia(dMass* m, uint index) {
//   if(!phy.copy_inertia)
//     return;
//   
//   for(uint i=0;i<9;i++)
//     m->I[i] = mujoco_inertia[index*9+i];
// }

void HumanoidWorld::apply_damping(dBodyID body, double v) {
  if(phy.damping == 1)
    dBodySetLinearDampingThreshold(body, v);
  else if(phy.damping == 2)
    dBodySetAngularDampingThreshold(body, v);
  else if(phy.damping == 3)
    dBodySetLinearDamping(body, v);
  else if(phy.damping == 4)
    dBodySetAngularDamping(body, v);
  else if(phy.damping == 5){
    dBodySetLinearDamping(body, v);
    dBodySetLinearDampingThreshold(body, v);
  } else if(phy.damping == 6) {
    dBodySetAngularDamping(body, v);
    dBodySetAngularDampingThreshold(body, v);
  }
}

void HumanoidWorld::createWorld() {
  ground = ODEFactory::getInstance()->createGround(odeworld);

//   dWorldSetCFM(odeworld.world_id, 1.);
//   dWorldSetERP(odeworld.world_id, 1.);

//   <compiler angle="degree" inertiafromgeom="true"/>
  double density = 943;  // so mass sum = 39.645

//   <joint armature="1" damping="1" limited="true"/>

//   armature
//     Armature inertia (or rotor inertia) of all degrees of freedom created by this joint. These are constants added to the diagonal of the inertia matrix in generalized coordinates. They make the simulation more stable, and often increase physical realism. This is because when a motor is attached to the system with a transmission that amplifies the motor force by c, the inertia of the rotor (i.e. the moving part of the motor) is amplified by c*c. The same holds for gears in the early stages of planetary gear boxes. These extra inertias often dominate the inertias of the robot parts that are represented explicitly in the model, and the armature attribute is the way to model them.
//   stiffness
//   A positive value generates a spring force (linear in position) acting along the tendon. The equilibrium length of the spring corresponds to the tendon length when the model is in its initial configuration.
//   solreflimit, solimplimit
//   Constraint solver parameters for simulating tendon limits. See Solver parameters.

//   <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
//   <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
//   done in collision

  double old_body_pos[3] = {0, 0, 0};
//   <body name="torso" pos="0 0 1.4">
  dBodyID torso = dBodyCreate(odeworld.world_id);
  old_body_pos[2] += 1.4;
  dBodySetPosition(torso, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  double torso_body_pos[3] = {old_body_pos[0], old_body_pos[1], old_body_pos[2]};

//   <joint name='rootx' type='slide' pos='0 0 0' axis='1 0 0' limited='false' damping='0' armature='0' stiffness='0' />
//   <joint name='rootz' type='slide' pos='0 0 0' axis='0 0 1' limited='false' damping='0' armature='0' stiffness='0' />
//   <joint name='rooty' type='hinge' pos='0 0 0' axis='0 1 0' limited='false' damping='0' armature='0' stiffness='0' />

//   <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>


//   <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
  dGeomID g_torso1 = dCreateCapsule(odeworld.space_id, 0.07, 0.14f);
  dGeomSetBody(g_torso1, torso);
  int rot_direction = 1;
  dMatrix3 Rot_y;
  dRFromAxisAndAngle(Rot_y, 1, 0, 0, M_PI/2.f);
  dGeomSetOffsetRotation(g_torso1, Rot_y);

  dMass m_torso;
  dMassSetCapsule(&m_torso, density, rot_direction, 0.07, 0.14f);

//   <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
  dGeomID g_head = dCreateSphere(odeworld.space_id, 0.09f);
  dGeomSetBody(g_head, torso);
  dGeomSetOffsetPosition(g_head, 0, 0, 0.19);

  dMass m_head;
  dMassSetSphere(&m_head, density, 0.09f);

//   <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
  dGeomID g_uwaist = dCreateCapsule(odeworld.space_id, 0.06f, 0.12f);
  dGeomSetBody(g_uwaist, torso);
  dGeomSetOffsetRotation(g_uwaist, Rot_y);
  dGeomSetOffsetPosition(g_uwaist, -.01, 0, -0.12);

  dMass m_uwaist;
  dMassSetCapsule(&m_uwaist, density, rot_direction, 0.06, 0.12f);

  dMassAdd(&m_torso, &m_head);
  dMassAdd(&m_torso, &m_uwaist);
//   copy_inertia(&m_torso, 0);
  dMassAdjust(&m_torso, 8.3220789393593630478562772623263299465179443359375);
  dBodySetMass(torso, &m_torso);

//   <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
  dBodyID lwaist = dBodyCreate(odeworld.world_id);
  old_body_pos[0] += -0.01;
  old_body_pos[2] += -0.260;
  dBodySetPosition(lwaist, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  dQuaternion q_lwaist = {1., 0, -0.002, 0};
  dBodySetQuaternion(lwaist, q_lwaist);

//   <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>
  dGeomID g_lwaist = dCreateCapsule(odeworld.space_id, 0.06f, 0.12f);
  dGeomSetBody(g_lwaist, lwaist);
  dGeomSetOffsetRotation(g_lwaist, Rot_y);

  dMass m_lwaist;
  dMassSetCapsule(&m_lwaist, density, rot_direction, 0.06, 0.12f);
  apply_armature(&m_lwaist, 0.02f);
  dMassAdjust(&m_lwaist, 2.035752039526185885875975145609118044376373291015625);
  dBodySetMass(lwaist, &m_lwaist);

//   <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
//   <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
  dJointID j_abdomen_zy = dJointCreateUniversal(odeworld.world_id, nullptr);
  dJointAttach(j_abdomen_zy, lwaist, torso);
  dJointSetUniversalAxis2(j_abdomen_zy, 0, 1, 0);//no body attached no effect
  dJointSetUniversalAxis1(j_abdomen_zy, 0, 0, 1);//no body attached no effect
  dJointSetUniversalParam(j_abdomen_zy, dParamLoStop, -45*M_PI/180.f);
  dJointSetUniversalParam(j_abdomen_zy, dParamHiStop, 45*M_PI/180.f);
  dJointSetUniversalParam(j_abdomen_zy, dParamLoStop2, -75*M_PI/180.f);
  dJointSetUniversalParam(j_abdomen_zy, dParamHiStop2, 30*M_PI/180.f);
  dJointSetUniversalAnchor(j_abdomen_zy, old_body_pos[0], old_body_pos[1], old_body_pos[2] + 0.065);
  apply_damping(lwaist, 5);

//   <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
  dBodyID pelvis = dBodyCreate(odeworld.world_id);
  old_body_pos[2] += -0.165;
  dBodySetPosition(pelvis, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  dBodySetQuaternion(pelvis, q_lwaist);
  double pelvis_body_pos[3] = {old_body_pos[0], old_body_pos[1], old_body_pos[2]};

//   <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
  dGeomID g_butt = dCreateCapsule(odeworld.space_id, 0.09f, 0.14f);
  dGeomSetBody(g_butt, pelvis);
  dGeomSetOffsetRotation(g_butt, Rot_y);
  dGeomSetOffsetPosition(g_butt, -0.02, 0, 0);

  dMass m_pelvis;
  dMassSetCapsule(&m_pelvis, density, rot_direction, 0.09, 0.14);
  apply_armature(&m_pelvis, 0.02f);
  dMassAdjust(&m_pelvis, 5.852787113637784699449184699915349483489990234375);
  dBodySetMass(pelvis, &m_pelvis);

//   <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
  dJointID j_abdomen_x = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_abdomen_x, pelvis, lwaist);
  dJointSetHingeAxis(j_abdomen_x, 1, 0, 0);//no body attached no effect
  dJointSetHingeParam(j_abdomen_x, dParamLoStop, -35*M_PI/180.f);
  dJointSetHingeParam(j_abdomen_x, dParamHiStop, 35*M_PI/180.f);
  dJointSetHingeAnchor(j_abdomen_x, old_body_pos[0], old_body_pos[1], old_body_pos[2] + 0.1f);
  apply_damping(pelvis, 5);
  
//   <body name="right_thigh" pos="0 -0.1 -0.04">
  dBodyID right_thigh = dBodyCreate(odeworld.world_id);
  old_body_pos[1] += -0.1;
  old_body_pos[2] += -0.04;
  dBodySetPosition(right_thigh, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

//   <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
  dGeomID g_right_thigh1 = dCreateCapsule(odeworld.space_id, 0.06f, sqrt(0.01*0.01 + 0.34*0.34));
  dGeomSetBody(g_right_thigh1, right_thigh);

  dMatrix3 R_right_thigh1;
  dRFromAxisAndAngle(R_right_thigh1, -1, 0, 0, 3.11219f);

  dGeomSetOffsetRotation(g_right_thigh1, R_right_thigh1);
  dGeomSetOffsetPosition(g_right_thigh1, 0., 0.01/2.f, -0.34/2.f);

  dMass m_right_thigh;
  dMassSetCapsule(&m_right_thigh, density, 3, 0.06, sqrt(0.01*0.01 + 0.34*0.34));
  apply_armature(&m_right_thigh, 0.01f);
  dMassAdjust(&m_right_thigh, 4.52555625774777592340569754014723002910614013671875);
  dBodySetMass(right_thigh, &m_right_thigh);

//   <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
//   <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
//   <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
  dJointID j_right_hip_xyz2 = dJointCreateBall(odeworld.world_id, nullptr);
  dJointAttach(j_right_hip_xyz2, pelvis, right_thigh);
  dJointSetBallAnchor(j_right_hip_xyz2, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  
  dJointID j_right_hip_xyz = dJointCreateAMotor(odeworld.world_id, nullptr);
  dJointAttach(j_right_hip_xyz, pelvis, right_thigh);
  dJointSetAMotorNumAxes(j_right_hip_xyz, 3);
  dJointSetAMotorMode(j_right_hip_xyz, dAMotorEuler);
  dJointSetAMotorAxis(j_right_hip_xyz, 0,1,0,-1,0);
  dJointSetAMotorAxis(j_right_hip_xyz, 2,2,0,0,-1);
  //[-25 5] axis x PI/2 ok
  //[-110 20] axis y out of PI/2
  //[-60 35] axis z in PI/2
  // axis z anchor to body 2
  // axis x or y anchored to body 1
  // axis y out of PI/2 so it must be controled
  dJointSetAMotorParam(j_right_hip_xyz, dParamLoStop3, -25*M_PI/180.f);
  dJointSetAMotorParam(j_right_hip_xyz, dParamHiStop3, 5*M_PI/180.f);
  dJointSetAMotorParam(j_right_hip_xyz, dParamLoStop, -110*M_PI/180.f);
  dJointSetAMotorParam(j_right_hip_xyz, dParamHiStop, 20*M_PI/180.f);
  dJointSetAMotorParam(j_right_hip_xyz, dParamLoStop2, -60*M_PI/180.f);
  dJointSetAMotorParam(j_right_hip_xyz, dParamHiStop2, 35*M_PI/180.f);
  apply_damping(right_thigh, 5);

//   <body name="right_shin" pos="0 0.01 -0.403">
  dBodyID right_shin = dBodyCreate(odeworld.world_id);
  old_body_pos[1] += 0.01;
  old_body_pos[2] += -0.403;
  dBodySetPosition(right_shin, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(right_shin, 1);

//   <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
  dGeomID g_right_shin1 = dCreateCapsule(odeworld.space_id, 0.049f, .3);
  dGeomSetBody(g_right_shin1, right_shin);
  dGeomSetOffsetPosition(g_right_shin1, 0,0, -.3/2.f);

  dMass m_right_shin;
  apply_armature(&m_right_shin, 0.006f);
  dMassSetCapsule(&m_right_shin, density, 3, 0.049, .3);

//   <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
  dJointID j_right_knee = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_right_knee, right_thigh, right_shin);
  dJointSetHingeAxis(j_right_knee, 0, 1, 0);//ignore negative axis
  dJointSetHingeParam(j_right_knee, dParamHiStop, -2*M_PI/180.f);
  dJointSetHingeParam(j_right_knee, dParamLoStop, -160*M_PI/180.f);//160 seems to bug (knee does one loop)
  dJointSetHingeAnchor(j_right_knee, old_body_pos[0], old_body_pos[1], old_body_pos[2]+ .02);

//   <body name="right_foot" pos="0 0 -0.45">
//   <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
  dGeomID g_right_foot = dCreateSphere(odeworld.space_id, 0.075f);
  dGeomSetBody(g_right_foot, right_shin);
  dGeomSetOffsetPosition(g_right_foot, 0, 0, -0.45f + 0.1f);
  
  dMass m_right_foot;
  dMassSetSphere(&m_right_foot, density, 0.075f);
  dMassAdd(&m_right_shin, &m_right_foot);
  dMassAdjust(&m_right_shin, 2.632494422482913432048690083320252597332000732421875 + 1.7671458676442586277488544510561041533946990966796875);
  dBodySetMass(right_shin, &m_right_shin);

//   <body name="left_thigh" pos="0 0.1 -0.04">
  dBodyID left_thigh = dBodyCreate(odeworld.world_id);
  old_body_pos[0] = pelvis_body_pos[0];
  old_body_pos[1] = pelvis_body_pos[1];
  old_body_pos[2] = pelvis_body_pos[2];
  old_body_pos[1] += 0.1;
  old_body_pos[2] += -0.04;
  dBodySetPosition(left_thigh, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

//   <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
  dGeomID g_left_thigh1 = dCreateCapsule(odeworld.space_id, 0.06f, sqrt(0.01*0.01 + 0.34*0.34));
  dGeomSetBody(g_left_thigh1, left_thigh);
  dMatrix3 R_left_thigh1;
  dRFromAxisAndAngle(R_left_thigh1, 1, 0, 0, 3.11219f);
  dGeomSetOffsetRotation(g_left_thigh1, R_left_thigh1);
  dGeomSetOffsetPosition(g_left_thigh1, 0., 0.01/2.f, -0.34/2.f);
  
  dMass m_left_thigh;
  dMassSetCapsule(&m_left_thigh, density, 3, 0.06, sqrt(0.01*0.01 + 0.34*0.34));
  apply_armature(&m_left_thigh, 0.01f);
  dMassAdjust(&m_left_thigh, 4.52555625774777592340569754014723002910614013671875);
  dBodySetMass(left_thigh, &m_left_thigh);

//   <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
//   <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-120 20" stiffness="20" type="hinge"/>
//   <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
  dJointID j_left_hip_xyz2 = dJointCreateBall(odeworld.world_id, nullptr);
  dJointAttach(j_left_hip_xyz2, pelvis, left_thigh);
  dJointSetBallAnchor(j_left_hip_xyz2, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

  dJointID j_left_hip_xyz = dJointCreateAMotor(odeworld.world_id, nullptr);
  dJointAttach(j_left_hip_xyz, pelvis, left_thigh);
  dJointSetAMotorMode(j_left_hip_xyz, dAMotorEuler);
  dJointSetAMotorAxis(j_left_hip_xyz, 0,1,0,-1,0);
  dJointSetAMotorAxis(j_left_hip_xyz, 2,2,0,0,1);
  dJointSetAMotorParam(j_left_hip_xyz, dParamLoStop3, -25*M_PI/180.f);
  dJointSetAMotorParam(j_left_hip_xyz, dParamHiStop3, 5*M_PI/180.f);
  dJointSetAMotorParam(j_left_hip_xyz, dParamLoStop, -110*M_PI/180.f);//right leg is set to 110
  dJointSetAMotorParam(j_left_hip_xyz, dParamHiStop, 20*M_PI/180.f);
  dJointSetAMotorParam(j_left_hip_xyz, dParamLoStop2, -60*M_PI/180.f);
  dJointSetAMotorParam(j_left_hip_xyz, dParamHiStop2, 35*M_PI/180.f);
  apply_damping(left_thigh, 5);

//   <body name="left_shin" pos="0 -0.01 -0.403">
  dBodyID left_shin = dBodyCreate(odeworld.world_id);
  old_body_pos[1] += -0.01;
  old_body_pos[2] += -0.403;
  dBodySetPosition(left_shin, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(left_shin, 1);

//   <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
  dGeomID g_left_shin1 = dCreateCapsule(odeworld.space_id, 0.049f, .3);
  dGeomSetBody(g_left_shin1, left_shin);
  dGeomSetOffsetPosition(g_left_shin1, 0,0, -.3/2.f);
  
  dMass m_left_shin;
  apply_armature(&m_left_shin, 0.006f);
  dMassSetCapsule(&m_left_shin, density, 3, 0.049, .3);

//  <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
  dJointID j_left_knee = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_left_knee, left_thigh, left_shin);
  dJointSetHingeAxis(j_left_knee, 0, 1, 0);//ignore negative axis
  dJointSetHingeParam(j_left_knee, dParamHiStop, -2*M_PI/180.f);
  dJointSetHingeParam(j_left_knee, dParamLoStop, -150*M_PI/180.f);//160 seems to bug (knee does one loop)
  dJointSetHingeAnchor(j_left_knee, old_body_pos[0], old_body_pos[1], old_body_pos[2]+ .02);

//   <body name="left_foot" pos="0 0 -0.45">
//   <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" />
  dGeomID g_left_foot = dCreateSphere(odeworld.space_id, 0.075f);
  dGeomSetBody(g_left_foot, left_shin);
  dGeomSetOffsetPosition(g_left_foot, 0, 0, -0.45f + 0.1f);
  
  dMass m_left_foot;
  dMassSetSphere(&m_left_foot, density, 0.075f);
  dMassAdd(&m_left_shin, &m_left_foot);
  dMassAdjust(&m_left_shin, 2.632494422482913432048690083320252597332000732421875 + 1.7671458676442586277488544510561041533946990966796875);
  dBodySetMass(left_shin, &m_left_shin);

//   <body name="right_upper_arm" pos="0 -0.17 0.06">
  dBodyID right_upper_arm = dBodyCreate(odeworld.world_id);
  old_body_pos[0] = torso_body_pos[0];
  old_body_pos[1] = torso_body_pos[1];
  old_body_pos[2] = torso_body_pos[2];
  old_body_pos[1] += -0.17;
  old_body_pos[2] += 0.06;
  dBodySetPosition(right_upper_arm, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(right_upper_arm, 1);

//   <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
//   dGeomID g_right_uarm1 = dCreateCapsule(odeworld.space_id, 0.04f, .16*2.);
  dGeomID g_right_uarm1 = dCreateCapsule(odeworld.space_id, 0.04f, sqrt(3*0.16*0.16));
  dGeomSetBody(g_right_uarm1, right_upper_arm);
  dMatrix3 R_right_uarm1;
  dRFromAxisAndAngle(R_right_uarm1, -1, -1, 0, M_PI/4.f);
  dGeomSetOffsetRotation(g_right_uarm1, R_right_uarm1);
  dGeomSetOffsetPosition(g_right_uarm1, 0.16/2.f, -0.16/2.f, -0.16/2.f);
  
  dMass m_right_uarm;
  dMassSetCapsule(&m_right_uarm, density, 3, 0.04f, sqrt(3*0.16*0.16));
  apply_armature(&m_right_uarm, 0.0068f);
  dMassAdjust(&m_right_uarm, 1.59405984156162627840558343450538814067840576171875);
  dBodySetMass(right_upper_arm, &m_right_uarm);

//   <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
//   <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
  dJointID j_right_shoulder = dJointCreateUniversal(odeworld.world_id, nullptr);
  dJointAttach(j_right_shoulder, right_upper_arm, torso);
  dJointSetUniversalAxis1(j_right_shoulder, 2, 1, 1);//no body attached no effect
  dJointSetUniversalAxis2(j_right_shoulder, 0, -1, 1);//no body attached no effect
  dJointSetUniversalParam(j_right_shoulder, dParamLoStop, -85*M_PI/180.f);
  dJointSetUniversalParam(j_right_shoulder, dParamHiStop, 60*M_PI/180.f);
  dJointSetUniversalParam(j_right_shoulder, dParamLoStop2, -85*M_PI/180.f);
  dJointSetUniversalParam(j_right_shoulder, dParamHiStop2, 60*M_PI/180.f);
  dJointSetUniversalAnchor(j_right_shoulder, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

//   <body name="right_lower_arm" pos=".18 -.18 -.18">
  dBodyID right_lower_arm = dBodyCreate(odeworld.world_id);
  old_body_pos[0] += 0.18;
  old_body_pos[1] += -0.18;
  old_body_pos[2] += -0.18;
  dBodySetPosition(right_lower_arm, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(right_lower_arm, 1);

//   <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
  dGeomID g_right_larm = dCreateCapsule(odeworld.space_id, 0.031f, sqrt(.16*.16*3.));
  dGeomSetBody(g_right_larm, right_lower_arm);
  dMatrix3 R_right_larm;
  dRFromAxisAndAngle(R_right_larm, -1, 1, 0, M_PI/4.f);
  dGeomSetOffsetRotation(g_right_larm, R_right_larm);
  dGeomSetOffsetPosition(g_right_larm, 0.18/2.f, 0.18/2.f, 0.18/2.f);
  
  dMass m_right_larm;
  apply_armature(&m_right_larm, 0.0028f);
  dMassSetCapsule(&m_right_larm, density, 3, 0.031, sqrt(.16*.16*3.));

//   <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
  dJointID j_right_elbow = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_right_elbow, right_lower_arm, right_upper_arm);
  dJointSetHingeAxis(j_right_elbow, 0, -1, 1);
  dJointSetHingeParam(j_right_elbow, dParamHiStop, 50*M_PI/180.);
  dJointSetHingeParam(j_right_elbow, dParamLoStop, -90*M_PI/180.);
  dJointSetHingeAnchor(j_right_elbow, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

//   <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
  dGeomID g_right_hand = dCreateSphere(odeworld.space_id, 0.04);
  dGeomSetBody(g_right_hand, right_lower_arm);
  dGeomSetOffsetPosition(g_right_hand, .18-0.02, .18-0.02, 0.18);
  
  dMass m_right_hand;
  dMassSetSphere(&m_right_hand, density, 0.04f);
  dMassAdd(&m_right_larm, &m_right_hand);
  dMassAdjust(&m_right_larm, 1.1983431305833824875861637337948195636272430419921875);
  dBodySetMass(right_lower_arm, &m_right_larm);

  //   <body name="left_upper_arm" pos="0 0.17 0.06">
  dBodyID left_upper_arm = dBodyCreate(odeworld.world_id);
  old_body_pos[0] = torso_body_pos[0];
  old_body_pos[1] = torso_body_pos[1];
  old_body_pos[2] = torso_body_pos[2];
  old_body_pos[1] += 0.17;
  old_body_pos[2] += 0.06;
  dBodySetPosition(left_upper_arm, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(left_upper_arm, 1);

  //   <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
  dGeomID g_left_uarm1 = dCreateCapsule(odeworld.space_id, 0.04f, sqrt(3*0.16*0.16));
  dGeomSetBody(g_left_uarm1, left_upper_arm);
  dMatrix3 R_left_uarm1;
  dRFromAxisAndAngle(R_left_uarm1, 1, -1, 0, M_PI/4.f);
  dGeomSetOffsetRotation(g_left_uarm1, R_left_uarm1);
  dGeomSetOffsetPosition(g_left_uarm1, 0.16/2.f, 0.16/2.f, -0.16/2.f);
  
  dMass m_left_uarm;
  dMassSetCapsule(&m_left_uarm, density, 3, 0.04f, sqrt(3*0.16*0.16));
  apply_armature(&m_left_uarm, 0.0068f);
  dMassAdjust(&m_left_uarm, 1.59405984156162627840558343450538814067840576171875);
  dBodySetMass(left_upper_arm, &m_left_uarm);

//   <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
//   <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
  dJointID j_left_shoulder = dJointCreateUniversal(odeworld.world_id, nullptr);
  dJointAttach(j_left_shoulder, left_upper_arm, torso);
  dJointSetUniversalAxis1(j_left_shoulder, 2, -1, 1);//no body attached no effect
  dJointSetUniversalAxis2(j_left_shoulder, 0, 1, 1);//no body attached no effect
  dJointSetUniversalParam(j_left_shoulder, dParamLoStop, -60*M_PI/180.f);
  dJointSetUniversalParam(j_left_shoulder, dParamHiStop, 85*M_PI/180.f);
  dJointSetUniversalParam(j_left_shoulder, dParamLoStop2, -60*M_PI/180.f);
  dJointSetUniversalParam(j_left_shoulder, dParamHiStop2, 85*M_PI/180.f);
  dJointSetUniversalAnchor(j_left_shoulder, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

  //   <body name="left_lower_arm" pos=".18 .18 -.18">
  dBodyID left_lower_arm = dBodyCreate(odeworld.world_id);
  old_body_pos[0] += 0.18;
  old_body_pos[1] += 0.18;
  old_body_pos[2] += -0.18;
  dBodySetPosition(left_lower_arm, old_body_pos[0], old_body_pos[1], old_body_pos[2]);
  apply_damping(left_lower_arm, 1);

//   <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
  dGeomID g_left_larm = dCreateCapsule(odeworld.space_id, 0.031f, sqrt(.16*.16*3.));
  dGeomSetBody(g_left_larm, left_lower_arm);
  dMatrix3 R_left_larm;
  dRFromAxisAndAngle(R_left_larm, 1, 1, 0, M_PI/4.f);
  dGeomSetOffsetRotation(g_left_larm, R_left_larm);
  dGeomSetOffsetPosition(g_left_larm, 0.18/2.f, -0.18/2.f, 0.18/2.f);

  dMass m_left_larm;
  apply_armature(&m_left_larm, 0.0028f);
  dMassSetCapsule(&m_left_larm, density, 3, 0.031, sqrt(.16*.16*3.));
  
  //   <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
  dJointID j_left_elbow = dJointCreateHinge(odeworld.world_id, nullptr);
  dJointAttach(j_left_elbow, left_lower_arm, left_upper_arm);
  dJointSetHingeAxis(j_left_elbow, 0, -1, -1);
  dJointSetHingeParam(j_left_elbow, dParamHiStop, 50*M_PI/180.);
  dJointSetHingeParam(j_left_elbow, dParamLoStop, -90*M_PI/180.);
  dJointSetHingeAnchor(j_left_elbow, old_body_pos[0], old_body_pos[1], old_body_pos[2]);

  //   <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
  dGeomID g_left_hand = dCreateSphere(odeworld.space_id, 0.04);
  dGeomSetBody(g_left_hand, left_lower_arm);
  dGeomSetOffsetPosition(g_left_hand, .18-0.02, -.18+0.02, 0.18);
  
  dMass m_left_hand;
  dMassSetSphere(&m_left_hand, density, 0.04f);
  dMassAdd(&m_left_larm, &m_left_hand);
  dMassAdjust(&m_left_larm, 1.1983431305833824875861637337948195636272430419921875);
  dBodySetMass(left_lower_arm, &m_left_larm);

  joints.push_back(j_abdomen_zy);
  joints.push_back(j_abdomen_x);
  joints.push_back(j_right_hip_xyz);
  joints.push_back(j_right_hip_xyz2);
  joints.push_back(j_right_knee);
  joints.push_back(j_left_hip_xyz);
  joints.push_back(j_left_hip_xyz2);
  joints.push_back(j_left_knee);
  joints.push_back(j_right_shoulder);
  joints.push_back(j_right_elbow);
  joints.push_back(j_left_shoulder);
  joints.push_back(j_left_elbow);

  bones.push_back(new ODEObject(torso, m_torso, g_torso1, 0,0,0.,density, m_torso.mass));
  bones.push_back(new ODEObject(nullptr, m_torso, g_head, 0,0,0.,density, m_head.mass));
  bones.push_back(new ODEObject(nullptr, m_torso, g_uwaist, 0,0,0.,density, m_uwaist.mass));

  bones.push_back(new ODEObject(lwaist, m_torso, g_lwaist, 0,0,0.,density, m_lwaist.mass));
  bones.push_back(new ODEObject(pelvis, m_torso, g_butt, 0,0,0.,density, m_pelvis.mass));//4
  bones.push_back(new ODEObject(right_thigh, m_torso, g_right_thigh1, 0,0,0.,density, m_right_thigh.mass));
  bones.push_back(new ODEObject(right_shin, m_torso, g_right_shin1, 0,0,0.,density, m_right_shin.mass));
  bones.push_back(new ODEObject(nullptr, m_torso, g_right_foot, 0,0,0.,density, m_right_foot.mass));
  bones.push_back(new ODEObject(left_thigh, m_torso, g_left_thigh1, 0,0,0.,density, m_left_thigh.mass));
  bones.push_back(new ODEObject(left_shin, m_torso, g_left_shin1, 0,0,0.,density, m_left_shin.mass));
  bones.push_back(new ODEObject(nullptr, m_torso, g_left_foot, 0,0,0.,density, m_left_foot.mass));

  bones.push_back(new ODEObject(right_upper_arm, m_torso, g_right_uarm1, 0,0,0.,density, m_right_uarm.mass));
  bones.push_back(new ODEObject(right_lower_arm, m_torso, g_right_larm, 0,0,0.,density, m_right_larm.mass));//12
  bones.push_back(new ODEObject(nullptr, m_torso, g_right_hand, 0,0,0.,density, m_right_hand.mass));//13

  bones.push_back(new ODEObject(left_upper_arm, m_torso, g_left_uarm1, 0,0,0.,density, m_left_uarm.mass));
  bones.push_back(new ODEObject(left_lower_arm, m_torso, g_left_larm, 0,0,0.,density, m_left_larm.mass));//15
  bones.push_back(new ODEObject(nullptr, m_torso, g_left_hand, 0,0,0.,density, m_left_hand.mass));//16
  
//   dJointID fixed_head = dJointCreateSlider(odeworld.world_id, nullptr);
//   dJointAttach(fixed_head, torso, nullptr);
// //   dJointSetSliderAxis(fixed_head, 1, 1, 0);
//   joints.push_back(fixed_head);
}

void nearCallbackHumanoid(void* data, dGeomID o1, dGeomID o2) {
  nearCallbackDataHumanoid* d = reinterpret_cast<nearCallbackDataHumanoid*>(data);
  HumanoidWorld* inst = d->inst;

  // only collide things with the ground | only to debug with humanoid
//   if(o1 != inst->ground && o2 != inst->ground)
//     return;
  
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnected(b1, b2)){
    return;
  }

  if (int numc = dCollide (o1,o2,2,&inst->contact[0].geom,sizeof(dContact))) {
    for (int i=0; i<numc; i++) {
      dJointID c = dJointCreateContact (inst->odeworld.world_id,inst->odeworld.contactgroup,&inst->contact[i]);
      dJointAttach (c, dGeomGetBody(o1), dGeomGetBody(o2));
    }
  }
}

void HumanoidWorld::step(const vector<double>& _motors) {
  std::vector<double> motors(_motors);

  double quad_ctrl_cost = 0.f;
  
  for (auto a : motors)
    quad_ctrl_cost += a*a;
  quad_ctrl_cost = 0.05f * quad_ctrl_cost;
  //quad_ctrl_cost in [0 ; 0.85]
  //0.85 <= ALIVE_BONUS (=1) so it's always better to have a high cost control but stay alive
  reward = ALIVE_BONUS - quad_ctrl_cost;
  
  for(uint i=0; i < 17; i++)
    motors[i] = std::min(std::max((double)-1., motors[i]), (double)1.);
  
  if(phy.control == 0) {
    for(uint i=0; i < 17 ; i++)
      qfrc_actuator[i] = bib::Utils::transform(motors[i], -1, 1, -gears[i], gears[i]);
  } else if(phy.control==1) {
    std::vector<double> p_motor(17);
    for(uint i=0; i < 17 ; i++)
      p_motor[i] = 2.0f/M_PI * atan(-2.0f*internal_state[5+i] - 0.05 * internal_state[22+6+i]);
    
    for(uint i=0; i < 17; i++)
      qfrc_actuator[i] = gears[i] * std::min(std::max((double)-1., p_motor[i]+motors[i]), (double)1.);
  }
  
  for(uint i=0;i<FRAME_SKIP;i++){
    dJointAddUniversalTorques(joints[0], qfrc_actuator[0], qfrc_actuator[1]);
    dJointAddHingeTorque(joints[1], qfrc_actuator[2]);
    dJointAddAMotorTorques(joints[2], qfrc_actuator[3], qfrc_actuator[4], qfrc_actuator[5]);
    dJointAddHingeTorque(joints[4], qfrc_actuator[6]);
    dJointAddAMotorTorques(joints[5], qfrc_actuator[7], qfrc_actuator[8], qfrc_actuator[9]);
    dJointAddHingeTorque(joints[7], qfrc_actuator[10]);
    dJointAddUniversalTorques(joints[8], qfrc_actuator[11], qfrc_actuator[12]);
    dJointAddHingeTorque(joints[9], qfrc_actuator[13]);
    dJointAddUniversalTorques(joints[10], qfrc_actuator[14], qfrc_actuator[15]);
    dJointAddHingeTorque(joints[11], qfrc_actuator[16]);

#ifdef DEBUG_MOTOR_BY_MOTOR
    std::vector<double> factors(17);
    for(uint i=0;i<gears.size();i++)
        factors[i] = 0;
      
    uint index_c = 3;
    double factor_ = -1.f;
    factors[index_c] = factor_;
  //   factors[3+4] = 1;
    
    for(uint i=0; i<gears.size(); i++)
      gears[i] = gears[i] * factors[i];

    dJointAddUniversalTorques(joints[0], gears[0], gears[1]);
    dJointAddHingeTorque(joints[1], gears[2]);
    dJointAddAMotorTorques(joints[2], gears[3], gears[4], gears[5]);
    dJointAddHingeTorque(joints[4], gears[6]);
    dJointAddAMotorTorques(joints[5], gears[7], gears[8], gears[9]);
    dJointAddHingeTorque(joints[7], gears[10]);
    dJointAddUniversalTorques(joints[8], gears[11], gears[12]);
    dJointAddHingeTorque(joints[9], gears[13]);
    dJointAddUniversalTorques(joints[10], gears[14], gears[15]);
    dJointAddHingeTorque(joints[11], gears[16]);
#endif

    Mutex::scoped_lock lock(ODEFactory::getInstance()->wannaStep());
    nearCallbackDataHumanoid d = {this};
    dSpaceCollide(odeworld.space_id, &d, &nearCallbackHumanoid);
    dWorldStep(odeworld.world_id, WORLD_STEP);

    dJointGroupEmpty(odeworld.contactgroup);
    lock.release();
  }


  update_state();
}

void HumanoidWorld::update_state() {
  uint begin_index = 0;
  
  std::vector<double> qpos(1 + 4 + 17);
  qpos[begin_index++]=dBodyGetPosition(bones[0]->getID())[2];
  const dReal * q_root = dBodyGetQuaternion(bones[0]->getID());
  for(uint i=0;i<4;i++)
    qpos[begin_index++]=q_root[i];
  
  qpos[begin_index++] = dJointGetUniversalAngle1(joints[0]);
  qpos[begin_index++] = dJointGetUniversalAngle2(joints[0]);
  qpos[begin_index++] = dJointGetHingeAngle(joints[1]);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[2], 0);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[2], 1);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[2], 2);
  qpos[begin_index++] = dJointGetHingeAngle(joints[4]);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[5], 0);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[5], 1);
  qpos[begin_index++] = dJointGetAMotorAngle(joints[5], 2);
  qpos[begin_index++] = dJointGetHingeAngle(joints[7]);
  qpos[begin_index++] = dJointGetUniversalAngle1(joints[8]);
  qpos[begin_index++] = dJointGetUniversalAngle2(joints[8]);
  qpos[begin_index++] = dJointGetHingeAngle(joints[9]);
  qpos[begin_index++] = dJointGetUniversalAngle1(joints[10]);
  qpos[begin_index++] = dJointGetUniversalAngle2(joints[10]);
  qpos[begin_index++] = dJointGetHingeAngle(joints[11]);
  
  ASSERT(begin_index == 22, "wrong index");

//   bib::Logger::PRINT_ELEMENTS(qpos);
  begin_index=0;
  std::vector<double> qvel(3 + 3 + 17);
  const dReal * torsolv = dBodyGetLinearVel(bones[0]->getID());
  for(uint i=0;i<3;i++)
    qvel[begin_index++]=torsolv[i];
  const dReal * torsoav = dBodyGetAngularVel(bones[0]->getID());
  for(uint i=0;i<3;i++)
    qvel[begin_index++]=torsoav[i];
  
  qvel[begin_index++] = dJointGetUniversalAngle1Rate(joints[0]);
  qvel[begin_index++] = dJointGetUniversalAngle2Rate(joints[0]);
  qvel[begin_index++] = dJointGetHingeAngleRate(joints[1]);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[2], 0);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[2], 1);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[2], 2);
  qvel[begin_index++] = dJointGetHingeAngleRate(joints[4]);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[5], 0);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[5], 1);
  qvel[begin_index++] = dJointGetAMotorAngleRate(joints[5], 2);
  qvel[begin_index++] = dJointGetHingeAngleRate(joints[7]);
  qvel[begin_index++] = dJointGetUniversalAngle1Rate(joints[8]);
  qvel[begin_index++] = dJointGetUniversalAngle2Rate(joints[8]);
  qvel[begin_index++] = dJointGetHingeAngleRate(joints[9]);
  qvel[begin_index++] = dJointGetUniversalAngle1Rate(joints[10]);
  qvel[begin_index++] = dJointGetUniversalAngle2Rate(joints[10]);
  qvel[begin_index++] = dJointGetHingeAngleRate(joints[11]);
  
  ASSERT(begin_index == 23, "wrong index");
//   bib::Logger::PRINT_ELEMENTS(qvel);

  std::copy(qpos.begin(), qpos.end(), internal_state.begin());
  std::copy(qvel.begin(), qvel.end(), internal_state.begin() + qpos.size());
  
  if(phy.additional_sensors){
    begin_index=0;
    std::vector<double> cinert(10*11);
    for(auto m : body_mass)
      cinert[begin_index++] = m;
    for(auto b : bones)
      if(b->getID() != nullptr){
        dMass m;
        dBodyGetMass(b->getID(), &m);
        for(uint i=0;i<9;i++)
          cinert[begin_index++] = m.I[i];
      }
    
    ASSERT(110 == begin_index, "pb");
    
    begin_index=0;
    std::vector<double> cvel(10*6);
    for (uint j=1;j < bones.size();j++) { //0 is already in qvel
      if(bones[j]->getID() != nullptr){
        const dReal * lv = dBodyGetLinearVel(bones[j]->getID());
        for(uint i=0;i<3;i++)
          cvel[begin_index++]=lv[i];
        const dReal * av = dBodyGetAngularVel(bones[j]->getID());
        for(uint i=0;i<3;i++)
          cvel[begin_index++]=av[i];
      }
    }
    
    ASSERT(60 == begin_index, "pb");
    
    //cfrc_ext is different than dBodyGetTorque/dBodyGetForce
    // in ode those value are always 0
    
    std::copy(cinert.begin(), cinert.end(), internal_state.begin() + qpos.size() + qvel.size());
    std::copy(cvel.begin(), cvel.end(), 
              internal_state.begin() + qpos.size() + qvel.size() + cinert.size());
    std::copy(qfrc_actuator.begin(), qfrc_actuator.end(), 
              internal_state.begin() + qpos.size() + qvel.size() + cinert.size() + cvel.size());
  }

  double lin_vel_cost = 0;
  double pos_after = mass_center();
  lin_vel_cost = (pos_after - pos_before) / WORLD_STEP;
  pos_before = pos_after;
  
//   it doesn't move enough and looks for a static rest
//   so remove factor 0.25
  lin_vel_cost = lin_vel_cost * phy.reward_scale_lvc;
  reward = reward + lin_vel_cost;
  
//   Another reward possibility?
//   std::vector<double> mc(body_mass.size());
//   begin_index = 0;
//   for(auto a : bones) {
//     if(a->getID() != nullptr){
//       auto pos = dBodyGetLinearVel(a->getID());
//       mc[begin_index] = pos[0] * body_mass[begin_index];
//       begin_index++;
//     }
//   }
//   
//   double np_sum = 0;
//   for(uint i=0;i < body_mass.size();i++)
//     np_sum += mc[i];
//   
//   reward = reward + (np_sum / mass_sum) * 0.25f;
}

double HumanoidWorld::mass_center(){
  //xipos in ODE the center of mass and the point of reference must coincide
  std::vector<double> mc(body_mass.size());
  
  uint begin_index = 0;
  for(auto a : bones) {
    if(a->getID() != nullptr){
      auto pos = dBodyGetPosition(a->getID());
      mc[begin_index] = pos[0] * body_mass[begin_index];
      begin_index++;
    }
  }
  ASSERT(begin_index == body_mass.size(), "pb");
  
  double np_sum = 0;
  for(uint i=0;i < body_mass.size();i++)
    np_sum += mc[i];
  
//   LOG_DEBUG("mass center " << (np_sum / mass_sum));
  return np_sum / mass_sum;
}

const std::vector<double>& HumanoidWorld::state() const {
  return internal_state;
}

unsigned int HumanoidWorld::activated_motors() const {
  return 17;
}

bool HumanoidWorld::final_state() const {
  //0.9 instead of 1.0 because of soft cfm
  return internal_state[0] < 0.9 || internal_state[0] > 2.0;
//   return false;
}

double HumanoidWorld::performance() const {
  return reward;
}

void HumanoidWorld::resetPositions(std::vector<double>&, const std::vector<double>&) {
//   LOG_DEBUG("resetPositions");

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

  createWorld();

  reward = ALIVE_BONUS;
  pos_before = 0.;

  std::fill(internal_state.begin(), internal_state.end(), 0.f);
  update_state();

//   LOG_DEBUG("endResetPositions");
}
