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

#include "arch/AACAgent.hpp"
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
  int replayed;

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
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }
    
    for (uint i = 0; i < a.size(); i++) {
      if(a[i] != b.a[i])
        return a[i] < b.a[i];
    }

    return false;
  }

} sample;

class OfflineCaclaAg : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl; 
   
  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors){

  }

  virtual ~OfflineCaclaAg() {
    delete vnn;
    delete ann;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning)
      trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, 0});

    last_pure_action.reset(new vector<double>(*next_action));
    //if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise){ //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    //}
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_v           = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a           = pt->get<int>("agent.hidden_unit_a");
    noise                   = pt->get<double>("agent.noise");
    update_pure_ac          = pt->get<bool>("agent.update_pure_ac");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    clear_trajectory        = pt->get<bool>("agent.clear_trajectory");
    update_delta_neg        = pt->get<bool>("agent.update_delta_neg");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    
    if(!gaussian_policy)
      noise = 0.05;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    if(clear_trajectory)
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
          double nextV = MLP::computeOutVF(local_nn, sm.next_s, {});
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

  void update_critic(){
      if (trajectory.size() > 0) {
        //remove trace of old policy

        std::vector<sample> vtraj(trajectory.size());
        std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());

        ParraVtoVNext dq(vtraj, this);

        auto iter = [&]() {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
          vnn->learn(dq.data, 10000, 0, 0.0001);
        };

        auto eval = [&]() {
          return fann_get_MSE(vnn->getNeuralNet());
        };

        if(determinist_vnn_update)
              bib::Converger::determinist<>(iter, eval, 30, 0.0001, 0);
        else {
          NN best_nn = nullptr;
          auto save_best = [&]() {
            if(best_nn != nullptr)
              fann_destroy(best_nn);
            best_nn = fann_copy(vnn->getNeuralNet());
          };

          bib::Converger::min_stochastic<>(iter, eval, save_best, 30, 0.0001, 0, 10);
          vnn->copy(best_nn);
          fann_destroy(best_nn);
        }

        dq.free();
      }
  }

  void end_episode() override {
    LOG_FILE("policy_exploration", ann->hash());
    
    if (trajectory.size() > 0) {

      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
        target = sm.r;
        if (!sm.goal_reached) {
          double nextV = vnn->computeOutVF(sm.next_s, {});
          target += gamma * nextV;
        }
        mine = vnn->computeOutVF(sm.s, {});

        if(target > mine) {
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          if(update_pure_ac){
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.pure_a[i];
          } else {
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.a[i];
          }

          n++;
        } else if(update_delta_neg && !update_pure_ac){
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.pure_a[i];
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
      
    update_critic();  
      
    if(!clear_trajectory){
      std::list<sample> current_trajectory;
      for(auto it = trajectory.begin(); it != trajectory.end() ; it++ ) {
        sample sa = *it;
        sa.replayed++;
        if(sa.replayed < 5)
          current_trajectory.push_back(sa);
      }
      trajectory.clear();
      
      trajectory.insert(current_trajectory.begin(), current_trajectory.end());
    }
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
    bib::XMLEngine::save<>(trajectory, "trajectory", path+".trajectory");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }
  
  double criticEval(const std::vector<double>& perceptions) override {
      return vnn->computeOutVF(perceptions, {});
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
        return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise);
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
  uint nb_sensors;
  
  bool update_pure_ac;

  double noise;
  bool gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update, clear_trajectory, update_delta_neg;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::set<sample> trajectory;
//     std::list<sample> trajectory;

  MLP* ann;
  MLP* vnn;
};

#endif

