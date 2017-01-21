#ifndef ODEOBJECT_H
#define ODEOBJECT_H
#include <ode/ode.h>
#include <bib/MyType.hpp>

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
  uint getColorMode() const;
  void setColorMode(uint);

 protected:
  dBodyID bid;
  dMass mass;
  dGeomID geom;
  double x, y, z, density, massv;
  uint color_mode=0;
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
