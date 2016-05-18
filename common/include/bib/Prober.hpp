#ifndef PROBER_H
#define PROBER_H

namespace bib {

class Prober {
 public:
  void probe(double m);

 public:
  double min_probe;
  double max_probe;
 protected:
  bool prob_init = false;
  long int prob_step = 0;
};
}  // namespace bib
#endif  // PROBER_H
