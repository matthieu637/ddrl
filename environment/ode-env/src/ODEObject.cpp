#include "ODEObject.hpp"
#include <bib/Utils.hpp>

ODEObject::ODEObject(dBodyID bid, dMass m, dGeomID geom, double x, double y,
                     double z, double density, double massv)
  : bid(bid),
    mass(m),
    geom(geom),
    x(x),
    y(y),
    z(z),
    density(density),
    massv(massv) {}

dBodyID ODEObject::getID() const {
  return bid;
}

dGeomID &ODEObject::getGeom() {
  return geom;
}

dMass &ODEObject::getMass() {
  return mass;
}

double ODEObject::getDensity() const {
  return density;
}

double ODEObject::getMassValue() const {
  return massv;
}

void ODEObject::setID(const dBodyID _id) {
  bid = _id;
}

double ODEObject::getX() const {
  return x;
}

double ODEObject::getY() const {
  return y;
}

double ODEObject::getZ() const {
  return z;
}

void ODEObject::setX(double nx) {
  x = nx;
}

void ODEObject::setY(double ny) {
  y = ny;
}

void ODEObject::setZ(double nz) {
  z = nz;
}

uint ODEObject::getColorMode() const{
  return color_mode;
}

void ODEObject::setColorMode(uint i){
  color_mode = i;
}

double ODEObject::distSince() {
  const dReal *pos = dGeomGetPosition(geom);
  return bib::Utils::euclidien_dist3D(x, pos[0], y, pos[1], z, pos[2]);
}

double ODEObject::distSinceX() {
  const dReal *pos = dGeomGetPosition(geom);
  return bib::Utils::euclidien_dist1D(x, pos[0]);
}

double ODEObject::distSinceY() {
  const dReal *pos = dGeomGetPosition(geom);
  return bib::Utils::euclidien_dist1D(y, pos[1]);
}

double ODEObject::distSinceZ() {
  const dReal *pos = dGeomGetPosition(geom);
  return bib::Utils::euclidien_dist1D(z, pos[2]);
}

ODEBox::ODEBox(dBodyID bid, dMass m, dGeomID geom, double x, double y, double z,
               double density, double mass, double lx, double ly, double lz)
  : ODEObject(bid, m, geom, x, y, z, density, mass), lx(lx), ly(ly), lz(lz) {}

double ODEBox::getLX() const {
  return lx;
}

double ODEBox::getLY() const {
  return ly;
}

double ODEBox::getLZ() const {
  return lz;
}

ODESphere::ODESphere(dBodyID bid, dMass m, dGeomID geom, double x, double y,
                     double z, double density, double massv, double radius)
  : ODEObject(bid, m, geom, x, y, z, density, massv), radius(radius) {}

double ODESphere::getRadius() const {
  return radius;
}
