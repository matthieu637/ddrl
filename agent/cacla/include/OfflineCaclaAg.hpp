#ifndef OFFLINECACLAAG_HPP
#define OFFLINECACLAAG_HPP

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
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(pure_a);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++)
      if (s[i] < b.s[i])
        return true;
    for (uint i = 0; i < pure_a.size(); i++)
      if (pure_a[i] < b.pure_a[i])
        return true;
    for (uint i = 0; i < a.size(); i++)
      if (a[i] < b.a[i])
        return true;
    for (uint i = 0; i < next_s.size(); i++)
      if (next_s[i] < b.next_s[i])
        return true;

    if (r < b.r)
      return true;

    return goal_reached < b.goal_reached;
  }

} sample;

class OfflineCaclaAg : public arch::AAgent<> {
 public:
  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~OfflineCaclaAg() {
    delete vnn;
    delete ann;
  }

  const std::vector<double>& run(double r, const std::vector<double>& sensors,
                                bool learning, bool goal_reached) override {

    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<double>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }


  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached) {

    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached) {
        double nextV = vnn->computeOut(sensors, {});
        vtarget += gamma * nextV;
      }
//             double lastv = vnn->computeOut(last_state, *next_action);

//             vnn->learn(last_state, {}, vtarget);

//             if (vtarget > lastv) //increase this action
//                 ann->learn(last_state, *last_action);

      trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});
    }

//         if (learning && bib::Utils::rand01() < alpha) { // alpha ??
//             for (uint i = 0; i < next_action->size(); i++)
//                 next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
//         }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
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
    gamma               = pt->get<double>("agent.gamma");
//         alpha_a               = pt->get<double>("agent.alpha_a");
    hidden_unit_v         = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a        = pt->get<int>("agent.hidden_unit_a");
    noise               = pt->get<double>("agent.noise");
    decision_each = pt->get<int>("agent.decision_each");

//         noise = 0.4;
//         hidden_unit_v = 25;
//         gamma = 0.99;
//         alpha_a = 0.05;
//         decision_each = 4;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors , 1, 0.0);
    else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors);
//         if (boost::filesystem::exists("trajectory.data")) {
//             decltype(trajectory)* obj = bib::XMLEngine::load<decltype(trajectory)>("trajectory", "trajectory.data");
//             trajectory = *obj;
//             delete obj;
//
//             end_episode();
//         }
  }

  void start_episode(const std::vector<double>& sensors) override {
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
    trajectory.clear();

    fann_reset_MSE(vnn->getNeuralNet());
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, const OfflineCaclaAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {
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
    const OfflineCaclaAg* ptr;
  };

  void removeOldPolicyTrajectory() {
    for(auto iter = trajectory.begin(); iter != trajectory.end(); ) {
      sample sm = *iter;
      vector<double> *ac = ann->computeOut(sm.s);

      bool eq = true;
      for(uint i = 0; i != sm.a.size(); i++)
        if (ac->at(i) != sm.a[i]) {
//                   LOG_DEBUG(ac->at(i) << " " << sm.a[i]);
          eq =false;
          break;
        }
//             if (*ac != sm.a) {
      if (!eq) {
        trajectory.erase(iter++);
      } else {
        ++iter;
      }
    }
  }

  void removeOldPolicyTrajectory2() {
    for(auto iter = trajectory.begin(); iter != trajectory.end(); ) {
      sample sm = *iter;
      vector<double> *ac = ann->computeOut(sm.s);

      bool eq = true;
      for(uint i = 0; i != sm.pure_a.size(); i++)
        if (ac->at(i) != sm.pure_a[i]) {
//                   LOG_DEBUG(ac->at(i) << " " << sm.a[i]);
          eq =false;
          break;
        }
//             if (*ac != sm.a) {
      if (!eq) {
        trajectory.erase(iter++);
      } else {
        ++iter;
      }
    }
  }

  void end_episode() override {

    if (trajectory.size() > 0) {
      //remove trace of old policy

      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());

      ParraVtoVNext dq(vtraj, this);

      auto iter = [&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);

        fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
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
//             bib::Converger::determinist<>(iter, eval, 30, 0.0001, 0);
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
      //       LOG_DEBUG("number of data " << trajectory.size());


      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        double target = sm.r;
        if (!sm.goal_reached) {
          double nextV = vnn->computeOut(sm.next_s, {});
          target += gamma * nextV;
        }

        double mine = vnn->computeOut(sm.s, {});

        if(target > mine) {
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          for (uint i = 0; i < nb_motors; i++)
            data->output[n][i] = sm.pure_a[i];
//                     should explain why
//                            data->output[n][i] = sm.a[i];

          n++;
        }
      }

      if(n > 0) {
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);

        ann->learn(subdata, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
    }

//         removeOldPolicyTrajectory();
//     removeOldPolicyTrajectory2();
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size();
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

  uint internal_time;
  uint decision_each;

  double gamma, noise;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  std::set<sample> trajectory;
//     std::list<sample> trajectory;

  MLP* ann;
  MLP* vnn;
};

#endif

