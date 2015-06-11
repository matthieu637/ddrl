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

typedef struct _sample {
    std::vector<float> s;
//     std::vector<float> a;
    std::vector<float> next_s;
    double r;
    bool goal_reached;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& BOOST_SERIALIZATION_NVP(s);
//         ar& BOOST_SERIALIZATION_NVP(a);
        ar& BOOST_SERIALIZATION_NVP(next_s);
        ar& BOOST_SERIALIZATION_NVP(r);
        ar& BOOST_SERIALIZATION_NVP(goal_reached);
    }

    bool operator< (const _sample& b) const {
        for (uint i = 0; i < s.size(); i++)
            if (s[i] < b.s[i])
                return true;
//         for (uint i = 0; i < a.size(); i++)
//             if (a[i] < b.a[i])
//                 return true;
        for (uint i = 0; i < next_s.size(); i++)
            if (next_s[i] < b.next_s[i])
                return true;

        if (r < b.r)
            return true;

        return goal_reached < b.goal_reached;
    }

} sample;

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
        if (reward >= 1.f) {
            reward = 13000;
//             reward = 1;
        } else {
            reward = exp((reward - 0.01) * 4000) * 0.01;
//             reward = 0;
        }
        internal_time ++;

        weighted_reward += reward * pow_gamma;
        pow_gamma *= gamma;

        sum_weighted_reward += reward * global_pow_gamma;
        global_pow_gamma *= gamma;

        time_for_ac--;
        if (time_for_ac == 0 || goal_reached) {
            const std::vector<float>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
//             time_for_ac = bib::Utils::transform(next_action[nb_motors], -1., 1., min_ac_time, max_ac_time);
            time_for_ac =4;

            for (uint i = 0; i < nb_motors; i++)
                returned_ac[i] = next_action[i];

            weighted_reward = 0;
            pow_gamma = 1.f;
        }

        return returned_ac;
    }

    double noise = 0.4;
    const std::vector<float>& _run(float reward, const std::vector<float>& sensors,
                                   bool learning, bool goal_reached) {

        vector<float>* next_action = ann->computeOut(sensors);

        if (last_action.get() != nullptr && learning) {  // Update Q

            double vtarget = reward;
            if (!goal_reached) {
//             if(aware_ac_time)
//               nn->learn(last_state, *last_action, reward + pow(gamma, bib::Utils::transform(last_action->at(last_action->size()-1),-1.,1., min_ac_time, max_ac_time) ) * nextQ);
//             else

                double nextV = vnn->computeOut(sensors, {});
                vtarget += gamma * nextV;
            }
            double lastv = vnn->computeOut(last_state, *next_action);

            vnn->learn(last_state, {}, vtarget);

            if (vtarget > lastv) //increase this action
                ann->learn(last_state, *last_action);

//             trajectory.insert( {last_state, *last_action, sensors, reward, goal_reached});
            trajectory.insert( {last_state, sensors, reward, goal_reached});
        }

//         if (learning && bib::Utils::rand01() < alpha) {
//             for (uint i = 0; i < next_action->size(); i++)
//                 next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
//         }
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

    
    void _unique_invoke(boost::property_tree::ptree*, boost::program_options::variables_map*) override {
//         epsilon             = pt->get<float>("agent.epsilon");
//         gamma               = pt->get<float>("agent.gamma");
//         alpha               = pt->get<float>("agent.alpha");
//         hidden_unit         = pt->get<int>("agent.hidden_unit");
// //     rlparam->activation          = pt->get<std::string>("agent.activation_function_hidden");
// //     rlparam->activation_stepness = pt->get<float>("agent.activation_steepness_hidden");
// //
// //     rlparam->repeat_replay = pt->get<int>("agent.replay");
// //
// //     int action_per_motor   = pt->get<int>("agent.action_per_motor");
// //
// //     sml::ActionFactory::getInstance()->gridAction(nb_motors, action_per_motor);
// //     actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());
// //
// //     act_templ = new sml::ActionTemplate( {"effectors"}, {sml::ActionFactory::getInstance()->getActionsNumber()});
// //     ainit = new sml::DAction(act_templ, {0});
// //     algo = new sml::QLearning<EnvState>(act_templ, *rlparam, nb_sensors);
        hidden_unit = 25;
        gamma = 0.99; // < 0.99  => gamma ^ 2000 = 0 && gamma != 1 -> better to reach the goal at the very end
//         gamma = 1.0d;
        //check 0,0099×((1−0.95^1999)÷(1−0.95))
        //r_max_no_goal×((1−gamma^1999)÷(1−gamma)) < r_max_goal * gamma^2000 && gamma^2000 != 0
        alpha = 0.05;
        epsilon = 0.1;

        min_ac_time = 4;
        max_ac_time = 4;

        aware_ac_time = false;

        vnn = new MLP(nb_sensors, hidden_unit, nb_sensors, alpha);
        ann = new LinMLP(nb_sensors , nb_motors);
//         if (boost::filesystem::exists("trajectory.data")) {
//             decltype(trajectory)* obj = bib::XMLEngine::load<decltype(trajectory)>("trajectory", "trajectory.data");
//             trajectory = *obj;
//             delete obj;
// 
//             end_episode();
//         }
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
        trajectory.clear();
        
//         noise = 0.99998*noise;
        fann_reset_MSE(vnn->getNeuralNet());
    }

    struct ParraVtoVNext {
        ParraVtoVNext(const std::vector<sample>& _vtraj, const BaseCaclaAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {
            data = fann_create_train(vtraj.size(), ptr->nb_sensors, 1);
            for (uint n = 0; n < vtraj.size(); n++) {
                sample sm = vtraj[n];
                for (uint i = 0; i < ptr->nb_sensors ; i++)
                    data->input[n][i] = sm.s[i];
            }
        }

        ~ParraVtoVNext() { //must be empty cause of tbb

        }

        void free() {
            fann_destroy_train(data);
        }

        void operator()(const tbb::blocked_range<size_t>& range) const {

            struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

            for (size_t n = range.begin(); n < range.end(); n++) {
                sample sm = vtraj[n];

                double delta = sm.r;
                if (!sm.goal_reached) {
                    double nextV = MLP::computeOut(local_nn, sm.next_s, {});
                    delta += ptr->gamma * nextV;
                }

                data->output[n][0] = delta;
            }

            fann_destroy(local_nn);
        }

        struct fann_train_data* data;
        const std::vector<sample>& vtraj;
        const BaseCaclaAg* ptr;
    };

    void end_episode() override {
//       if (trajectory.size() > 0) {
//             std::vector<sample> vtraj(trajectory.size());
//             std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
// 
//             ParraVtoVNext dq(vtraj, this);
// 
//             auto iter = [&]() {
//               tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);
// 
//       //         fann_randomize_weights(nn->getNeuralNet(), -0.025, 0.025);
//               vnn->learn(dq.data);
// 
//       //                 uint number_par = 6;
//       //                 ParrallelLearnFromScratch plfs(dq.data, nn->getNeuralNet(), number_par);
//       //                 tbb::parallel_for(tbb::blocked_range<uint>(0, number_par), plfs);
//       //                 nn->copy(plfs.bestNN());
//       //                 plfs.free();
//             };
// 
//             auto eval = [&]() {
//               return fann_get_MSE(vnn->getNeuralNet());
//             };
// 
//       //             bib::Converger::determinist<>(iter, eval, 250, 0.0005, 1);
// 
//             NN best_nn = nullptr;
//             auto save_best = [&]() {
//               best_nn = fann_copy(vnn->getNeuralNet());
//             };
// 
//             bib::Converger::min_stochastic<>(iter, eval, save_best, 100, 0.0001, 0, 20);
//             vnn->copy(best_nn);
//             fann_destroy(best_nn);
// 
//             dq.free();
//       //       LOG_DEBUG("number of data " << trajectory.size());
//           }
    }

    void save(const std::string& path) override {
        ann->save(path+".actor");
        vnn->save(path+".critic");
        bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
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
    std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }

private:
    uint nb_motors;
    uint nb_sensors;
    uint time_for_ac;

    double weighted_reward;
    double pow_gamma;
    double global_pow_gamma;
    double sum_weighted_reward;

    uint min_ac_time;
    uint max_ac_time;

    uint internal_time;

    bool aware_ac_time;

    double epsilon, alpha, gamma;
    uint hidden_unit;

    std::shared_ptr<std::vector<float>> last_action;
    std::vector<float> last_state;

    std::vector<float> returned_ac;

    std::set<sample> trajectory;
//     std::list<sample> trajectory;

    MLP* ann;
    MLP* vnn;
};

#endif

