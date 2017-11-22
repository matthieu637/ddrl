#ifndef HALTCHEETAHWORLDVIEW_HPP
#define HALTCHEETAHWORLDVIEW_HPP

#include <string>
#include <list>
#include <vector>
#include "tbb/tbb.h"

#include "drawstuff.h"
#include "HalfCheetahWorld.hpp"

class HalfCheetahWorldView : public HalfCheetahWorld {
 public:
  HalfCheetahWorldView(const std::string &path, const hcheetah_physics phy, bool capture);
  ~HalfCheetahWorldView();
  void step(const std::vector<double> &motors) override;
  void step_core(const std::vector<double> &motors) override;
  void resetPositions(std::vector<double> & result_stoch, const std::vector<double>& given_stoch) override;

protected:
  virtual void createWorld() override;
  
 public:
  //     std::list<dGeomID> geoms;
  //     std::list<ODEObject*> delete_me_later;

  tbb::tbb_thread *eventThread;
  dsFunctions fn;
  bool requestEnd;
  double capture;

  // specific keyboard behavior
  double speed;
  bool ignoreMotor;
  double modified_motor = -2.f;
  
  Mutex mutex_reset;
};

#endif  // HALTCHEETAHWORLDVIEW_HPP
