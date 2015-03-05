#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <vector>

#include "arch/AAgent.hpp"
#include "bib/Utils.hpp"

class PowerAg : public arch::AAgent<> {
 public:
  PowerAg(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<float>& run(float, const std::vector<float>&, bool, bool) {
    for (unsigned int i = 0; i < actuator.size(); i++)
      actuator[i] = bib::Utils::randin(-1, 1);
    return actuator;
  }

  std::vector<float> actuator;
};

#endif  // EXAMPLE_H
