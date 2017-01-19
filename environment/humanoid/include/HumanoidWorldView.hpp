#ifndef HALTCHEETAHWORLDVIEW_HPP
#define HALTCHEETAHWORLDVIEW_HPP

#include <string>
#include <list>
#include <vector>
#include "tbb/tbb.h"

#include "drawstuff.h"
#include "HumanoidWorld.hpp"

class HumanoidWorldView : public HumanoidWorld {
 public:
  HumanoidWorldView(const std::string &path, const humanoid_physics phy);
  ~HumanoidWorldView();
  void step(const std::vector<double> &motors) override;

 public:
  //     std::list<dGeomID> geoms;
  //     std::list<ODEObject*> delete_me_later;

  tbb::tbb_thread *eventThread;
  dsFunctions fn;
  bool requestEnd;

  // specific keyboard behavior
  double speed;
  bool ignoreMotor;
  double modified_motor = -2.f;
  
  Mutex mutex_reset;
};

#endif  // HALTCHEETAHWORLDVIEW_HPP