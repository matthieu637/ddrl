#ifndef ODEFACTORY_H
#define ODEFACTORY_H

#include <list>
#include <memory>
#include "ode/ode.h"
#include "tbb/queuing_mutex.h"

#include "bib/Singleton.hpp"
#include "bib/Logger.hpp"
#include "ODEObject.hpp"


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
  ODEObject *createBox(const ODEWorld &world, double x, double y, double z,
                       double lx, double ly, double lz, double density,
                       bool linkBody = true, double inertia=-1.f);
  ODEObject *createSphere(const ODEWorld &world, double x, double y, double z,
                          double radius, double density,
                          bool linkBody = true);

  ODEWorld createWorld();
  void destroyWorld(const ODEWorld &);

  Mutex &wannaStep();

  static dGeomID createGround(const ODEWorld &world);

 protected:
  ODEFactory();
 public:
  ~ODEFactory();

 protected:
#ifdef NO_PARALLEL
  // only to display them in single thread impl
  std::list<dGeomID> createdGeom;
#endif

  Mutex mutex, mutexStep;
};

#endif  // ODEFACTORY_H
