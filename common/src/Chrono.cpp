#include "bib/Chrono.hpp"

using namespace std::chrono;

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
} // namespace bib
