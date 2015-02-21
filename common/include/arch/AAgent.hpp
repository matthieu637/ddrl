#ifndef AAGENT_H
#define AAGENT_H

#include "arch/Dummy.hpp"

namespace arch {

template <typename ProgOptions = AgentProgOptions>
class AAgent : public ProgOptions
{
public:
    virtual const std::vector<float>& run(float reward, const std::vector<float>&, bool, bool) = 0;

    virtual void unique_invoke(boost::property_tree::ptree*) {

    }

    virtual void start_episode(const std::vector<float>&) {

    }

    std::ostream& display(std::ostream& out, bool display, bool dump) {
        if(display) {

        }

        if(dump) {

        }

        return out;
    }
};

}

#endif // AAGENT_H
