#ifndef DISCRETACTION_H
#define DISCRETACTION_H

#include "arch/AAgent.hpp"
#include "sml/QLearning.hpp"
#include <vector>
#include "sml/ActionFactory.hpp"


#define NUMBER_OF_DISCRET_ACTION

typedef boost::shared_ptr<std::vector<float>> EnvState;

class DiscretActionAg : public arch::AAgent<> {
  public:
    DiscretActionAg(unsigned int _nb_motors, unsigned int _nb_sensors) : nb_motors(_nb_motors), nb_sensors(_nb_sensors) {
        ainit = nullptr;
        actions = nullptr;
        algo = nullptr;
        act_templ = nullptr;
    }

    ~DiscretActionAg() {
        delete ainit;
        delete actions;
        delete algo;
        delete act_templ;

        sml::ActionFactory::endInstance();
    }

    const std::vector<float>& run(float reward, const std::vector<float>& sensors, bool learning, bool goal_reached) {
        EnvState s(new std::vector<float>(sensors));
        sml::DAction* ac;
        if (learning)
            ac = algo->learn(s, reward, goal_reached);
        else ac = algo->decision(s, false);

        vector<float>* outputs = sml::ActionFactory::computeOutputs(ac, 0, *actions);
        if (!learning)
            delete ac;
        return *outputs;
    }

    void unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) {
        rlparam.epsilon = pt->get<float>("agent.epsilon");
        rlparam.gamma = pt->get<float>("agent.gamma");

        rlparam.alpha = pt->get<float>("agent.alpha");
        rlparam.hidden_unit = pt->get<int>("agent.hidden_unit");
        rlparam.activation = pt->get<std::string>("agent.activation_function_hidden");
        rlparam.activation_stepness = pt->get<float>("agent.activation_steepness_hidden");

        rlparam.activation_stepness = pt->get<int>("agent.replay");

        int number_discret_action = pt->get<int>("agent.discret_action");

        sml::ActionFactory::getInstance()->injectArgs(number_discret_action);
        sml::ActionFactory::getInstance()->randomFixedAction(nb_motors, 1, 1);
        actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());

        act_templ = new sml::ActionTemplate( {"effectors"}, {number_discret_action});
        ainit = new sml::DAction(act_templ, {0});
        algo = new sml::QLearning<EnvState>(act_templ, rlparam, nb_sensors);
    }

    void start_episode(const std::vector<float>& sensors) {
        EnvState s(new std::vector<float>(sensors));
        algo->startEpisode(s, *ainit);
    }

  private:
    int nb_motors;
    int nb_sensors;

    sml::QLearning<EnvState>* algo;
    sml::ActionTemplate* act_templ;
    sml::list_tlaction* actions;
    sml::DAction* ainit;
    sml::RLParam rlparam;
};

#endif // DISCRETACTION_H
