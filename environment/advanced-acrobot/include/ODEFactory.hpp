#ifndef ODEFACTORY_H
#define ODEFACTORY_H

#include "bib/Singleton.hpp"
#include "bib/Logger.hpp"
#include "ODEObject.hpp"
#include <ode/ode.h>
#include <list>
#include <memory>
#include <tbb/queuing_mutex.h>

typedef std::shared_ptr<ODEObject> pODEObject;
typedef tbb::queuing_mutex Mutex;

struct ODEWorld {
  dWorldID world_id;
  dSpaceID space_id;
  dJointGroupID contactgroup;
};

class ODEFactory : public bib::Singleton<ODEFactory> {
  friend class bib::Singleton<ODEFactory>;

 public:
  ODEObject *createBox(const ODEWorld &world, float x, float y, float z,
                       float lx, float ly, float lz, float density, float mass,
                       bool linkBody = true);
  ODEObject *createSphere(const ODEWorld &world, float x, float y, float z,
                          float radius, float density, float mass,
                          bool linkBody = true);

  ODEWorld createWorld();
  void destroyWorld(const ODEWorld &);

  Mutex &wannaStep();

  dGeomID createGround(const ODEWorld &world);

 protected:
  ODEFactory();
  ~ODEFactory();

 protected:
#ifdef NO_PARALLEL
  // only to display them in single thread impl
  std::list<dGeomID> createdGeom;
#endif

  Mutex mutex, mutexStep;
};

#endif  // ODEFACTORY_H
