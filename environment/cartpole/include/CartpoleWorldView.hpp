#ifndef CARTPOLEWORLDVIEW_HPP
#define CARTPOLEWORLDVIEW_HPP

#include <string>
#include <list>
#include <vector>
#include "tbb/tbb.h"

#include "drawstuff.h"
#include "CartpoleWorld.hpp"

class CartpoleWorldView : public CartpoleWorld {
 public:
  CartpoleWorldView(const std::string &,
                           bool add_time_in_state = false,
                           bool normalization = false,
                           const std::vector<double>& normalized_vector = {}
                          );
  ~CartpoleWorldView();
  void step(const std::vector<double> &motors, uint current_step, uint max_step_per_instance);

 public:
  std::list<dGeomID> geoms;
  //     std::list<ODEObject*> delete_me_later;

  tbb::tbb_thread *eventThread;
  dsFunctions fn;
  bool requestEnd;

  // specific keyboard behavior
  double speed;
  bool ignoreMotor;
  double modified_motor = -2.f;
};

#endif  // CARTPOLEWORLDVIEW_HPP
