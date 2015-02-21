
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"

class ExampleEnv : public arch::AEnvironment<> {
public:

    const std::vector<float>& perceptions() const
    {
        return internal_state;
    }

    float performance() {
        return 0;
    }
    std::vector<float> internal_state;
};


class ExampleAgent : public arch::AAgent<> {
public:
    const std::vector<float>& run(float, const std::vector<float>&, bool, bool) {
        return actuator;
    }
    
    std::vector<float> actuator;
};


int main(int argc, char **argv)
{
    arch::Simulator<ExampleEnv, ExampleAgent> s;
    s.init(argc, argv);

    s.run();

    LOG_DEBUG("works !");
}
