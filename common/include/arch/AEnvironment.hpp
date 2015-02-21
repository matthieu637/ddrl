#ifndef AENVIRONMENT_H
#define AENVIRONMENT_H

#include "arch/Dummy.hpp"
#include "arch/CommonAE.hpp"

namespace arch {

template <typename ProgOptions = DummyProgOptions>
class AEnvironment : public ProgOptions, public CommonAE
{
public:
    virtual void unique_destroy() {
    }

    virtual const std::vector<float>& perceptions() const = 0;


    virtual float performance() = 0 ;
    
    virtual ~AEnvironment(){
      //pure?
    };

    void unique_invoke(boost::property_tree::ptree* properties) {
        instance_per_episode = properties->get<unsigned int>("environment.instance_per_episode");
        max_step_per_instance = properties->get<unsigned int>("environment.max_step_per_instance");
    }

    void reset_episode() {
        current_instance = 0;
        current_step = 0;
    }

    void next_instance() {
        current_step = 0;
        current_instance++;
    }

    void apply(const std::vector<float>&) {
        current_step++;
    }

    bool running() const {
        return current_step < max_step_per_instance;
    }

    bool hasInstance() const {
        return current_instance < instance_per_episode;
    }

    unsigned int current_step=0;
    unsigned int current_instance=0;

    unsigned int max_step_per_instance;
    unsigned int instance_per_episode;
};

}

#endif // AENVIRONMENT_H
