#include "Draw.hpp"

#include <string>
#include <list>

std::vector<ODEObject *> *Draw::geoms = nullptr;

void Draw::drawGeom(dGeomID g) {
  if (!g) return;
  
  const dReal *pos = dGeomGetPosition(g);
  const dReal *R = dGeomGetRotation(g);
  
  int type = dGeomGetClass(g);
  if (type == dBoxClass) {
    dVector3 sides;
    dGeomBoxGetLengths(g, sides);
    dsDrawBox(pos, R, sides);
  } else if (type == dSphereClass) {
    dsDrawSphere(pos, R, dGeomSphereGetRadius(g));
  } else if (type == dCapsuleClass) {
    dReal radius, length;
    dGeomCapsuleGetParams(g, &radius, &length);
    dsDrawCapsule(pos, R, length, radius);
  } else if (type == dCylinderClass) {
    //----> Convex Object
    dReal radius, length;
    dGeomCylinderGetParams(g, &radius, &length);
    dsDrawCylinder(pos, R, length, radius);
  }
//   else if (type == dGeomTransformClass) {
//     dGeomID g2 = dGeomTransformGetGeom(g);
//     const dReal *pos2 = dGeomGetPosition(g2);
//     const dReal *R2 = dGeomGetRotation(g2);
//     dVector3 actual_pos;
//     dMatrix3 actual_R;
//     dMultiply0_331(actual_pos, R, pos2);
//     actual_pos[0] += pos[0];
//     actual_pos[1] += pos[1];
//     actual_pos[2] += pos[2];
//     dMultiply0_333(actual_R, R, R2);
//     drawGeom(g2, actual_pos, actual_R);
//   }
  if (false) {
    dBodyID body = dGeomGetBody(g);
    if (body) {
      const dReal *bodypos = dBodyGetPosition(body);
      const dReal *bodyr = dBodyGetRotation(body);
      dReal bodySides[3] = {0.1, 0.1, 0.1};
      dsSetColorAlpha(0, 1, 0, 1);
      dsDrawBox(bodypos, bodyr, bodySides);
    }
  }
}

void Draw::drawLoop(int) {
  // draw world trimesh

  if (Draw::geoms != nullptr) {
    auto list = Draw::geoms;
    for (auto it = list->begin(); it != list->end(); ++it) {
      if ((*it)->getColorMode() == 1) {
        dsSetTexture(DS_CHECKERED);
        dsSetColor(255, 0., 0.);
      } else {
        dsSetColor(0.7, 0.7, 0.4);
        dsSetTexture(DS_WOOD);
      }
      drawGeom((*it)->getGeom());
    }
  }
}
