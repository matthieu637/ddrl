#ifndef ADVANCEDACROBOTWORLDVIEW_HPP
#define ADVANCEDACROBOTWORLDVIEW_HPP

#include <string>
#include "tbb/tbb.h"

#include "drawstuff.h"
#include "AdvancedAcrobotWorld.hpp"

class AdvancedAcrobotWorldView : public AdvancedAcrobotWorld {
 public:
  AdvancedAcrobotWorldView(const std::string &,
                           const std::vector<bone_joint> &types = {HINGE,
                                                                   HINGE
                                                                  },
                           const std::vector<bool> &actuators = {false, false,
                                                                 true
                                                                });
  ~AdvancedAcrobotWorldView();
  void step(const std::vector<float> &motors);

 public:
  std::list<dGeomID> geoms;
  //     std::list<ODEObject*> delete_me_later;

  tbb::tbb_thread *eventThread;
  dsFunctions fn;
  bool requestEnd;

  // specific keyboard behavior
  bool speedUp;
  bool ignoreMotor;
};

#endif  // ADVANCEDACROBOTWORLDVIEW_HPP
