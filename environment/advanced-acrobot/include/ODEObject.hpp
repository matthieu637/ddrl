#ifndef ODEOBJECT_H
#define ODEOBJECT_H
#include <ode/ode.h>

class ODEObject {
 public:
  ODEObject(dBodyID bid, dMass m, dGeomID geom, double x, double y, double z,
            double density, double massv);
  dBodyID getID() const;
  double getX() const;
  double getY() const;
  double getZ() const;
  dGeomID& getGeom();
  dMass& getMass();
  double getDensity() const;
  double getMassValue() const;
  void setID(const dBodyID getID);
  void setX(double nx);
  void setY(double ny);
  void setZ(double nz);
  double distSince();
  double distSinceX();
  double distSinceY();
  double distSinceZ();

 protected:
  dBodyID bid;
  dMass mass;
  dGeomID geom;
  double x, y, z, density, massv;
};

class ODEBox : public ODEObject {
 public:
  ODEBox(dBodyID bid, dMass m, dGeomID geom, double x, double y, double z,
         double density, double mass, double lx, double ly, double lz);
  double getLX() const;
  double getLY() const;
  double getLZ() const;

 protected:
  double lx, ly, lz;
};

class ODESphere : public ODEObject {
 public:
  ODESphere(dBodyID bid, dMass m, dGeomID geom, double x, double y, double z,
            double density, double massv, double radius);
  double getRadius() const;

 protected:
  double radius;
};

#endif  // ODEOBJECT_H
