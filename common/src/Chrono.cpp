#include "bib/Chrono.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace bib {

void Chrono::start() {
  begin = high_resolution_clock::now();
}
double Chrono::finish() {
  high_resolution_clock::time_point end = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(end - begin);
  return time_span.count();
}

void Chrono::reset() {
  start();
}
}  // namespace bib

