#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "arch/AAgent.hpp"
#include "arch/AEnvironment.hpp"
#include <bib/Utils.hpp>

namespace arch {

class ExampleEnv : public arch::AEnvironment<> {
public:
  
    ExampleEnv(){
      
    }

    const std::vector<float>& perceptions() const
    {
        return internal_state;
    }

    unsigned int number_of_actuators() const {
        return 3;
    }

    float performance() {
        return 0;
    }

    void _apply(const std::vector<float>&) {

    }

    std::vector<float> internal_state;
};


class ExampleAgent : public arch::AAgent<> {
public:
    ExampleAgent(unsigned int nb_motors): actuator(nb_motors) {
    }

    const std::vector<float>& run(float, const std::vector<float>&, bool, bool) {
        for(unsigned int i=0; i < actuator.size(); i++)
            actuator[i] = bib::Utils::randin(-1, 1);
        return actuator;
    }

    std::vector<float> actuator;
};

}

#endif // EXAMPLE_H
