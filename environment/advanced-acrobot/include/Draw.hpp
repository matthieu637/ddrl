#ifndef DRAW_HPP
#define DRAW_HPP

#include "drawstuff.h"
#include <ode/ode.h>

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawConvex dsDrawConvexD
#endif

#include "ODEFactory.hpp"


class Draw
{
public:
    static std::list<dGeomID>* geoms;
    
    static void drawGeom (dGeomID g, const dReal *pos, const dReal *R);
    
    static void drawLoop (int);
};

#endif
