#ifndef ODEOBJECT_H
#define ODEOBJECT_H
#include <ode/ode.h>

class ODEObject {
 public:
  ODEObject(dBodyID bid, dMass m, dGeomID geom, float x, float y, float z,
            float density, float massv);
  dBodyID getID() const;
  float getX() const;
  float getY() const;
  float getZ() const;
  dGeomID& getGeom() const;
  dMass& getMass() const;
  float getDensity() const;
  float getMassValue() const;
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
  float getLX() const;
  float getLY() const;
  float getLZ() const;

 protected:
  float lx, ly, lz;
};

class ODESphere : public ODEObject {
 public:
  ODESphere(dBodyID bid, dMass m, dGeomID geom, float x, float y, float z,
            float density, float massv, float radius);
  float getRadius() const;

 protected:
  float radius;
};

#endif  // ODEOBJECT_H
