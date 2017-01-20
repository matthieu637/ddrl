#ifndef DRAW_HPP
#define DRAW_HPP

#include <vector>
#include "ode/ode.h"

#include "drawstuff.h"

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawConvex dsDrawConvexD
#endif

#include "ODEFactory.hpp"

class Draw {
 public:
//   static std::list<dGeomID> *geoms;
  static std::vector<ODEObject *>* geoms;

  static void drawGeom(dGeomID g);

  static void drawLoop(int);
};

#endif
