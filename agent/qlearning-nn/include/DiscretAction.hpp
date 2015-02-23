#ifndef DISCRETACTION_H
#define DISCRETACTION_H

#include "arch/AAgent.hpp"
#include "sml/QLearning.hpp"
#include <vector>
#include "sml/ActionFactory.hpp"


typedef boost::shared_ptr<std::vector<float>> EnvState;

class DiscretAction : public arch::AAgent<>
{
public:
    DiscretAction(unsigned int nb_motors, unsigned int nb_sensors) {
	sml::ActionFactory::getInstance()->randomFixedAction(nb_motors, 1, 1);
	actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());
	
        act_templ = new sml::ActionTemplate( {"effectors"}, {(int) nb_motors});
	
	sml::RLParam param;
        algo = new sml::QLearning<EnvState>(act_templ, param, nb_sensors);
    }

    ~DiscretAction() {
        delete act_templ;
    }

    const std::vector<float>& run(float reward, const std::vector<float>& sensors, bool goal, bool) {
	EnvState s(new std::vector<float>(sensors));
	sml::DAction* ac = algo->learn(s, reward, goal);
	
        return *sml::ActionFactory::computeOutputs(ac, 1, *actions);
    }

private:
    sml::QLearning<EnvState>* algo;
    sml::ActionTemplate* act_templ;
    sml::list_tlaction* actions;
};

#endif // DISCRETACTION_H
