#ifndef ADVANCEDACROBOTWORLDVIEW_HPP
#define ADVANCEDACROBOTWORLDVIEW_HPP

#include <string>
#include <list>
#include <vector>
#include "tbb/tbb.h"

#include "drawstuff.h"
#include "AdvancedAcrobotWorld.hpp"

class AdvancedAcrobotWorldView : public AdvancedAcrobotWorld {
 public:
  AdvancedAcrobotWorldView(const std::string &,
                           const std::vector<bone_joint> &types = {HINGE, HINGE},
                           const std::vector<bool> &actuators = {false, false, true},
                           bool add_time_in_state = false,
                           bool normalization = false,
                           const std::vector<double>& normalized_vector = {}
                          );
  ~AdvancedAcrobotWorldView();
  void step(const std::vector<double> &motors, uint current_step, uint max_step_per_instance);

 public:
  //   std::list<dGeomID> geoms;
  //   std::list<ODEObject*> delete_me_later;

  tbb::tbb_thread *eventThread = nullptr;
  dsFunctions fn;
  bool requestEnd;
  bool changeThread;

  // specific keyboard behavior
  double speed;
  bool ignoreMotor;
};

#endif  // ADVANCEDACROBOTWORLDVIEW_HPP
