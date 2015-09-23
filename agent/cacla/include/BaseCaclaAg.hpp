#ifndef BASECACLAAG_HPP
#define BASECACLAAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/AAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "MLP.hpp"
#include "LinMLP.hpp"

class BaseCaclaAg : public arch::AAgent<>
{
public:
    BaseCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
        : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

    }

    virtual ~BaseCaclaAg() {
        delete vnn;
        delete ann;
    }

    const std::vector<float>& run(float r, const std::vector<float>& sensors,
                                  bool learning, bool goal_reached) override {

        double reward = r;
        internal_time ++;

        weighted_reward += reward * pow_gamma;
        pow_gamma *= gamma;

        sum_weighted_reward += reward * global_pow_gamma;
        global_pow_gamma *= gamma;

        time_for_ac--;
        if (time_for_ac == 0 || goal_reached) {
            const std::vector<float>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
            time_for_ac = decision_each;

            for (uint i = 0; i < nb_motors; i++)
                returned_ac[i] = next_action[i];

            weighted_reward = 0;
            pow_gamma = 1.f;
        }

        return returned_ac;
    }

    const std::vector<float>& _run(float reward, const std::vector<float>& sensors,
                                   bool learning, bool goal_reached) {

        vector<float>* next_action = ann->computeOut(sensors);

        if (last_action.get() != nullptr && learning) {  // Update Q

            double vtarget = reward;
            if (!goal_reached) {
                double nextV = vnn->computeOut(sensors, {});
                vtarget += gamma * nextV;
            }
            double lastv = vnn->computeOut(last_state, *next_action);

            vnn->learn(last_state, {}, vtarget);

            if (vtarget > lastv) //increase this action
                ann->learn(last_state, *last_action);
        }

        if(learning){
           vector<float>* randomized_action = bib::Proba<float>::multidimentionnalGaussianWReject(*next_action, noise);
           delete next_action;
           next_action = randomized_action;
        }
        last_action.reset(next_action);

        last_state.clear();
        for (uint i = 0; i < sensors.size(); i++)
            last_state.push_back(sensors[i]);

        return *next_action;
    }

    
    void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {

        gamma               = pt->get<float>("agent.gamma");
        alpha_v             = pt->get<float>("agent.alpha_v");
        alpha_a             = pt->get<float>("agent.alpha_a");
        noise               = pt->get<float>("agent.noise");
        hidden_unit         = pt->get<int>("agent.hidden_unit");
        decision_each       = pt->get<int>("agent.decision_each");

//         hidden_unit = 25;
//         gamma = 0.9; // < 0.99  => gamma ^ 2000 = 0 && gamma != 1 -> better to reach the goal at the very end
        //check 0,0099×((1−0.95^1999)÷(1−0.95))
        //r_max_no_goal×((1−gamma^1999)÷(1−gamma)) < r_max_goal * gamma^2000 && gamma^2000 != 0
//         alpha = 0.05;
//         noise = 0.4;

        vnn = new MLP(nb_sensors, hidden_unit, nb_sensors, alpha_v);
        ann = new LinMLP(nb_sensors , nb_motors, alpha_a);
    }

    void start_episode(const std::vector<float>& sensors) override {
        last_state.clear();
        for (uint i = 0; i < sensors.size(); i++)
            last_state.push_back(sensors[i]);

        last_action = nullptr;

        weighted_reward = 0;
        pow_gamma = 1.d;
        time_for_ac = 1;
        sum_weighted_reward = 0;
        global_pow_gamma = 1.f;
        internal_time = 0;
        
        fann_reset_MSE(vnn->getNeuralNet());
    }

    void end_episode() override {
      
    }

    void save(const std::string& path) override {
        ann->save(path+".actor");
        vnn->save(path+".critic");
    }

    void load(const std::string& path) override {
        ann->load(path+".actor");
        vnn->load(path+".critic");
    }

protected:
    void _display(std::ostream& out) const override {
        out << sum_weighted_reward << " " << std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise ;
    }
    
  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) << 
    sum_weighted_reward << " " << std::setw(8) << std::fixed << 
    std::setprecision(5) << vnn->error() ;
  }

private:
    uint nb_motors;
    uint nb_sensors;
    uint time_for_ac;

    double weighted_reward;
    double pow_gamma;
    double global_pow_gamma;
    double sum_weighted_reward;

    uint internal_time, decision_each;

    double alpha_v, gamma, alpha_a;
    double noise;
    uint hidden_unit;

    std::shared_ptr<std::vector<float>> last_action;
    std::vector<float> last_state;

    std::vector<float> returned_ac;

    MLP* ann;
    MLP* vnn;
};

#endif

