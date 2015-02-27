#ifndef CHRONO_H
#define CHRONO_H

#include <chrono>

using std::chrono::high_resolution_clock;

namespace bib {

class Chrono {
 public:
  void start();
  double finish();
  void reset();

 private:
  high_resolution_clock::time_point begin;
};
} // namespace bib
#endif  // CHRONO_H
