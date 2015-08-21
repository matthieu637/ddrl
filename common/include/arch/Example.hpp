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
  const std::vector<double>& perceptions() const {
    return internal_state;
  }
  unsigned int number_of_actuators() const {
    return 3;
  }
  unsigned int number_of_sensors() const {
    return internal_state.size();
  }
  double performance() const {
    return 0;
  }
  void _apply(const std::vector<double>&) {}

  std::vector<double> internal_state;
};

class ExampleAgent : public arch::AAgent<> {
 public:
  ExampleAgent(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<double>& run(double, const std::vector<double>&, bool, bool) override {
    for (unsigned int i = 0; i < actuator.size(); i++)
      actuator[i] = bib::Utils::randin(-1, 1);
    return actuator;
  }

  virtual ~ExampleAgent() {
  }

  std::vector<double> actuator;
};
}  // namespace arch

#endif  // EXAMPLE_H
