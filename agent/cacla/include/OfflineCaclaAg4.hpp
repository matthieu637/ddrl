#ifndef OFFLINECACLAAG3_HPP
#define OFFLINECACLAAG3_HPP

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
#include "Trajectory.hpp"

class OfflineCaclaAg : public arch::AAgent<> {
 public:
  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~OfflineCaclaAg() {
    delete trajectory_a;
    delete trajectory_v;
    
    delete vnn;
    delete ann;
  }

  const std::vector<float>& runf(float r, const std::vector<float>& sensors,
                                 bool learning, bool goal_reached, bool finished) override {
    if(r >= 1.) {
      noise = 0.0005;
    }
    
    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<float>& next_action = _runf(weighted_reward, sensors, learning, goal_reached, finished);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }


  const std::vector<float>& _runf(float reward, const std::vector<float>& sensors,
                                  bool learning, bool goal_reached, bool finished) {

    vector<float>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached) {
        double nextV = vnn->computeOut(sensors, *next_action);
        vtarget += gamma * nextV;
      }

      double mine = vnn->computeOut(last_state, *last_action);

      if (vtarget > mine && episode % 5 == 0 /*&& (!finished || goal_reached)*/) {//increase this action

//         trajectory_a->insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});
        ann->learn(last_state, *last_action);
        
        //error here cause only lake on policy not on value fonction - false Q(s,a) -> R + g * Q(sn, A)
//         if(!finished)
//           update_critic();
      }

      QSASRG_sample vsampl = {last_state, *last_action, sensors, reward, goal_reached};
//       QSASRG_sample vsampl = {last_state, *last_pure_action, sensors, reward, goal_reached};
      trajectory_v->addPoint(vsampl);
    }

//         if (learning && bib::Utils::rand01() < alpha) { // alpha ??
//             for (uint i = 0; i < next_action->size(); i++)
//                 next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
//         }

    last_pure_action.reset(new vector<float>(*next_action));
    if(learning) {
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
    alpha_a               = pt->get<float>("agent.alpha_a");
    hidden_unit_v         = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a        = pt->get<int>("agent.hidden_unit_a");
    noise               = pt->get<float>("agent.noise");
    decision_each = pt->get<int>("agent.decision_each");

    episode=0;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, alpha_a);
    else {
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors);
      fann_set_learning_rate(ann->getNeuralNet(), alpha_a);
    }

    trajectory_a = new Trajectory<SA_sample>(nb_sensors, 0.5f, false);
//     trajectory_v = new Trajectory<QSASRG_sample>(nb_sensors + nb_motors, 1.f, true, 1.5f);
    trajectory_v = new Trajectory<QSASRG_sample>(nb_sensors + nb_motors, 0.5f, true, 0.5f);
  }

  void start_episode(const std::vector<float>& sensors) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;
    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
//         trajectory_v->clear();
//         trajectory_a->clear();
    episode ++;

    fann_reset_MSE(vnn->getNeuralNet());
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<QSASRG_sample>& _vtraj, const OfflineCaclaAg* _ptr) : vtraj(_vtraj), ptr(_ptr),
      actions(vtraj.size()) {

      data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        QSASRG_sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];

        actions[n] = ptr->ann->computeOut(sm.next_s);
      }
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
      for (uint n = 0; n < vtraj.size(); n++) {
        delete actions[n];
      }
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        QSASRG_sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          double nextV = MLP::computeOut(local_nn, sm.next_s, *actions[n]);
          delta += ptr->gamma * nextV;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
    }

    struct fann_train_data* data;
    const std::vector<QSASRG_sample>& vtraj;
    const OfflineCaclaAg* ptr;
    std::vector<std::vector<float>*> actions;
  };

  void update_critic(){
    if (trajectory_v->size() > 0) {

      std::vector<QSASRG_sample> vtraj(trajectory_v->size());
      std::copy(trajectory_v->tree().begin(), trajectory_v->tree().end(), vtraj.begin());

      ParraVtoVNext dq(vtraj, this);

      auto iter = [&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);

//         fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
        vnn->learn(dq.data, 10000, 0, 0.0001);

        //                 uint number_par = 6;
        //                 ParrallelLearnFromScratch plfs(dq.data, nn->getNeuralNet(), number_par);
        //                 tbb::parallel_for(tbb::blocked_range<uint>(0, number_par), plfs);
        //                 nn->copy(plfs.bestNN());
        //                 plfs.free();
      };

      auto eval = [&]() {
        return fann_get_MSE(vnn->getNeuralNet());
      };

// CHOOSE BETWEEN determinist or stochastic (don't change many)
//       bib::Converger::determinist<>(iter, eval, 30, 0.0001, 0);
// OR
      NN best_nn = nullptr;
      auto save_best = [&]() {
        if(best_nn != nullptr)
          fann_destroy(best_nn);
        best_nn = fann_copy(vnn->getNeuralNet());
      };

      bib::Converger::min_stochastic<>(iter, eval, save_best, 30, 0.0001, 0, 10);
      vnn->copy(best_nn);
      fann_destroy(best_nn);
// END CHOOSE

      dq.free(); 
      
    }
  }
  
  void end_episode() override {
      update_critic();
// ACTION LEARNING
//             struct fann_train_data* data = fann_create_train(trajectory_a->size(), nb_sensors, nb_motors);
//
//             uint n=0;
//             for(auto it = trajectory_a->begin(); it != trajectory_a->end() ; ++it) {
//                 sample sm = *it;
//
//                 for (uint i = 0; i < nb_sensors ; i++)
//                     data->input[n][i] = sm.s[i];
//                 for (uint i = 0; i < nb_motors; i++)
// //                     data->output[n][i] = sm.pure_a[i];
// //                     should explain why
//                     data->output[n][i] = sm.a[i];
//
//                 n++;
//             }
//
//             if(n > 0) {
//                 struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);
//
//                 ann->learn(subdata, 5000, 0, 0.0001);
//
//                 fann_destroy_train(subdata);
//             }
//             fann_destroy_train(data);
//

//             trajectory_v->clear();
//             trajectory_a->clear();
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
//     bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " <<
        std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " <<
        trajectory_v->size() << " " << trajectory_a->size();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory_v->size() ;
  }

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;

  uint internal_time;
  uint decision_each;
  uint episode;

  double gamma, noise, alpha_a;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<float>> last_action;
  std::shared_ptr<std::vector<float>> last_pure_action;
  std::vector<float> last_state;

  std::vector<float> returned_ac;

  Trajectory<SA_sample>* trajectory_a;
  Trajectory<QSASRG_sample>* trajectory_v;

  MLP* ann;
  MLP* vnn;
};

#endif

