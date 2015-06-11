#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <vector>

#include "arch/AAgent.hpp"
#include "arch/AEnvironment.hpp"
#include "bib/Utils.hpp"

namespace arch {

class ExampleEnv : public arch::AEnvironment<> {
 public:
  ExampleEnv() : internal_state(6) {
    for (unsigned int i = 0; i < internal_state.size(); i++)
      internal_state[i] = bib::Utils::randin(-1, 1);
  }
  const std::vector<float>& perceptions() const {
    return internal_state;
  }
  unsigned int number_of_actuators() const {
    return 3;
  }
  unsigned int number_of_sensors() const {
    return internal_state.size();
  }
  float performance() const {
    return 0;
  }
  void _apply(const std::vector<float>&) {}

  std::vector<float> internal_state;
};

class ExampleAgent : public arch::AAgent<> {
 public:
  ExampleAgent(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<float>& run(float, const std::vector<float>&, bool, bool) override {
    for (unsigned int i = 0; i < actuator.size(); i++)
      actuator[i] = bib::Utils::randin(-1, 1);
    return actuator;
  }

  virtual ~ExampleAgent() {
  }

  std::vector<float> actuator;
};
}  // namespace arch

#endif  // EXAMPLE_H
