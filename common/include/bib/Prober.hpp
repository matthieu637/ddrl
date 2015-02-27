#ifndef PROBER_H
#define PROBER_H

namespace bib {

class Prober {
 public:
  void probe(float m);

 protected:
  float min_probe;
  float max_probe;
  bool prob_init = false;
  int prob_step = 0;
};
} // namespace bib
#endif  // PROBER_H
