#ifndef CONTINUOUSACAG_HPP
#define CONTINUOUSACAG_HPP

#include <vector>
#include <string>

#include "arch/AAgent.hpp"
#include <bib/Seed.hpp>
#include <bib/Utils.hpp>
#include "MLP.hpp"

class ContinuousAcAg : public arch::AAgent<>
{
public:
    ContinuousAcAg(unsigned int _nb_motors, unsigned int _nb_sensors)
        : nb_motors(_nb_motors), nb_sensors(_nb_sensors) {

    }

    ~ContinuousAcAg() {
        delete nn;
    }

    const std::vector<float>& run(float reward, const std::vector<float>& sensors,
                                  bool learning, bool goal_reached) override {
        vector<float>* next_action = nn->optimized(sensors);

        if (last_action.get() != nullptr && learning) { //Update Q
            double nextQ = nn->computeOut(sensors, *next_action);
            if (!goal_reached)
                nn->learn(last_state, *last_action, reward + gamma * nextQ);
            else
                nn->learn(last_state, *last_action, reward);
        }

        if (bib::Seed::getInstance()->rand() < alpha) {
            for (uint i = 0; i < next_action->size(); i++)
                next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
        }
        last_action.reset(next_action);

        last_state.clear();
        for (uint i = 0; i < sensors.size(); i++)
            last_state.push_back(sensors[i]);

        return *next_action;
    }

    void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
//     rlparam = new sml::RLParam;
        epsilon             = pt->get<float>("agent.epsilon");
        gamma               = pt->get<float>("agent.gamma");
//
        alpha               = pt->get<float>("agent.alpha");
        hidden_unit         = pt->get<int>("agent.hidden_unit");
//     rlparam->activation          = pt->get<std::string>("agent.activation_function_hidden");
//     rlparam->activation_stepness = pt->get<float>("agent.activation_steepness_hidden");
//
//     rlparam->repeat_replay = pt->get<int>("agent.replay");
//
//     int action_per_motor   = pt->get<int>("agent.action_per_motor");
//
//     sml::ActionFactory::getInstance()->gridAction(nb_motors, action_per_motor);
//     actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());
//
//     act_templ = new sml::ActionTemplate( {"effectors"}, {sml::ActionFactory::getInstance()->getActionsNumber()});
//     ainit = new sml::DAction(act_templ, {0});
//     algo = new sml::QLearning<EnvState>(act_templ, *rlparam, nb_sensors);
        hidden_unit=50;
        nn = new MLP(nb_sensors + nb_motors, hidden_unit, nb_sensors, alpha);
    }

    void start_episode(const std::vector<float>& sensors) override {
//     EnvState s(new std::vector<float>(sensors));
//     algo->startEpisode(s, *ainit);
//     weighted_reward = 0;
//     pow_gamma = 1.d;
        last_state.clear();
        for (uint i = 0; i < sensors.size(); i++)
            last_state.push_back(sensors[i]);
    }

    void end_episode() override {

    }

    void save(const std::string& path) override {

    }

    void load(const std::string& path) override {

    }

protected:
    void _display(std::ostream& stdout) const override {

    }

private:
    int nb_motors;
    int nb_sensors;

    double epsilon, alpha, gamma;
    uint hidden_unit;

    std::shared_ptr<std::vector<float>> last_action;
    std::vector<float> last_state;

    MLP* nn;
};

#endif  // DISCRETACTION_H
