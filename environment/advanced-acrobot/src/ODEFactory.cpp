#include "ODEFactory.hpp"
#include "bib/Logger.hpp"

ODEFactory::ODEFactory() {
  //     dInitODE2(dInitFlagManualThreadCleanup);
  dInitODE2(0);
}

ODEWorld ODEFactory::createWorld() {
  Mutex::scoped_lock lock(mutex);  // acquire
  //     LOG_DEBUG("call for create");
  //     dAllocateODEDataForThread(dAllocateFlagCollisionData);

  ODEWorld world;
  world.world_id = dWorldCreate();

  world.space_id = dHashSpaceCreate(0);
  //     dSpaceSetManualCleanup(world.space_id, 0);

  world.contactgroup = dJointGroupCreate(0);

  lock.release();

  return world;
}

void ODEFactory::destroyWorld(const ODEWorld &world) {
  Mutex::scoped_lock lock(mutex);
  //     LOG_DEBUG("call for destory");
  dJointGroupDestroy(world.contactgroup);

  dSpaceDestroy(world.space_id);
  dWorldDestroy(world.world_id);

  //     dCleanupODEAllDataForThread();
  lock.release();
}

Mutex &ODEFactory::wannaStep() {
  return mutex;
}

ODEFactory::~ODEFactory() {
  dCloseODE();
}

ODEObject *ODEFactory::createBox(const ODEWorld &world, float x, float y,
                                 float z, float lx, float ly, float lz,
                                 float density, bool linkBody) {
  dGeomID boxgeom = dCreateBox(world.space_id, lx, ly, lz);

  dMass m;
  dMassSetBox(&m, density, lx, ly, lz);

  dBodyID box_id;
  if (linkBody) {
    box_id = dBodyCreate(world.world_id);
    dBodySetPosition(box_id, x, y, z);

    dBodySetMass(box_id, &m);

    dGeomSetBody(boxgeom, box_id);
  } else {
    box_id = nullptr;
  }

  ODEObject *box = new ODEBox(box_id, m, boxgeom, x, y, z, density, m.mass, lx, ly, lz);
  return box;
}

ODEObject *ODEFactory::createSphere(const ODEWorld &world, float x, float y,
                                    float z, float radius, float density,
                                    bool linkBody) {
  dGeomID sphgeom = dCreateSphere(world.space_id, radius);

  dMass m;
  dMassSetSphere(&m, density, radius);

  dBodyID sphere_id;
  if (linkBody) {
    sphere_id = dBodyCreate(world.world_id);
    dBodySetPosition(sphere_id, x, y, z);

    dBodySetMass(sphere_id, &m);

    dGeomSetBody(sphgeom, sphere_id);
  } else {
    sphere_id = nullptr;
  }

  ODEObject *sphere =
    new ODESphere(sphere_id, m, sphgeom, x, y, z, density, m.mass, radius);
  return sphere;
}

dGeomID ODEFactory::createGround(const ODEWorld &world) {
  return dCreatePlane(world.space_id, 0, 0, 1, 0);
}
