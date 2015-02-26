#ifndef ODEOBJECT_H
#define ODEOBJECT_H
#include <ode/ode.h>

class ODEObject {
 public:
  ODEObject(dBodyID bid, dMass m, dGeomID geom, float x, float y, float z,
            float density, float massv);
  dBodyID getID();
  float getX();
  float getY();
  float getZ();
  dGeomID& getGeom();
  dMass& getMass();
  float getDensity();
  float getMassValue();
  void setID(const dBodyID getID);
  void setX(float nx);
  void setY(float ny);
  void setZ(double nz);
  float distSince();
  float distSinceX();
  float distSinceY();
  float distSinceZ();

 protected:
  dBodyID bid;
  dMass mass;
  dGeomID geom;
  float x, y, z, density, massv;
};

class ODEBox : public ODEObject {
 public:
  ODEBox(dBodyID bid, dMass m, dGeomID geom, float x, float y, float z,
         float density, float mass, float lx, float ly, float lz);
  float getLX();
  float getLY();
  float getLZ();

 protected:
  float lx, ly, lz;
};

class ODESphere : public ODEObject {
 public:
  ODESphere(dBodyID bid, dMass m, dGeomID geom, float x, float y, float z,
            float density, float massv, float radius);
  float getRadius();

 protected:
  float radius;
};

#endif  // ODEOBJECT_H
