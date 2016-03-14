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
#include <bib/Combinaison.hpp>
#include "MLP.hpp"
#include "LinMLP.hpp"
#include "Trajectory.hpp"
#include "../../qlearning-nn/include/MLP.hpp"

class HybridCaclaAg : public arch::AAgent<> {
 public:
  HybridCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~HybridCaclaAg() {
    delete trajectory_a;
    delete trajectory_v;
    
    delete vnn;
    delete ann;
  }

  const std::vector<double>& runf(double r, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool finished) override {

    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<double>& next_action = _runf(weighted_reward, sensors, learning, goal_reached, finished);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }


  const std::vector<double>& _runf(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached) {
        double nextV = vnn->computeOutVF(sensors, {});
        vtarget += gamma * nextV;
      }

      double mine = vnn->computeOutVF(last_state, {});
      sum_VS += mine;

      if (vtarget > mine /*&& (!finished || goal_reached)*/) {//increase this action
        
        SA_sample as;
        if(update_pure_ac)
          as = {last_state, *last_pure_action};
        else 
          as = {last_state, *last_action};
        
//         LOG_DEBUG(trajectory_a->size() << " " << trajectory_a->count_inserted << " " << trajectory_a->count_removed << " - " <<
//           trajectory_v->size() << " " << trajectory_v->count_inserted << " " << trajectory_v->count_removed 
//         );

        trajectory_a->addPoint(as);

        if(online_update){
          update_actor();
          removeOldPolicyTrajectory();
        }
        
//         LOG_DEBUG(trajectory_a->size() << " " << trajectory_a->count_inserted << " " << trajectory_a->count_removed << " - " <<
//           trajectory_v->size() << " " << trajectory_v->count_inserted << " " << trajectory_v->count_removed 
//         );
      }

      SASRG_sample vsampl = {last_state, *last_pure_action, sensors, reward, goal_reached};
      trajectory_v->addPoint(vsampl);
      
      if(online_update)
        update_critic();
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      }
      else {
      //e greedy
      if(bib::Utils::rand01() < noise)
        for (uint i = 0; i < next_action->size(); i++)
            next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

//         LOG_DEBUG(last_state[0] << " "<< last_state[1]<< " "<< last_state[2]<< " "<< last_state[3]<< " "<<
//              last_state[4]);
//     LOG_FILE("test_ech.data", last_state[0] << " "<< last_state[1]<< " "<< last_state[2]<< " "<< last_state[3]<< " "<<
//              last_state[4]);
    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    gamma               = pt->get<double>("agent.gamma");
//         alpha_a               = pt->get<double>("agent.alpha_a");
    hidden_unit_v       = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a       = pt->get<int>("agent.hidden_unit_a");
    noise               = pt->get<double>("agent.noise");
    decision_each       = pt->get<int>("agent.decision_each");
    lecun_activation    = pt->get<bool>("agent.lecun_activation");
    online_update       = pt->get<bool>("agent.online_update");
    update_pure_ac      = pt->get<bool>("agent.update_pure_ac");
    kd_tree_autoremove  = pt->get<bool>("agent.kd_tree_autoremove");
    clear_trajectory    = pt->get<bool>("agent.clear_trajectory");
    update_vtraj_actor  = pt->get<bool>("agent.update_vtraj_actor");
    transform_proba     = pt->get<double>("agent.transform_proba");
    gaussian_policy     = pt->get<bool>("agent.gaussian_policy");
    
    if(!gaussian_policy)
      noise = 0.05;

    episode=-1;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
    
    //need normalization of s
    trajectory_a = new Trajectory<SA_sample>(nb_sensors, transform_proba, kd_tree_autoremove);
    
    //don't need normalization if removeTrajectory
    trajectory_v = new Trajectory<SASRG_sample>(nb_sensors, transform_proba, kd_tree_autoremove);
    
  }

  void start_episode(const std::vector<double>& sensors, bool) override {
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
    
    
    sum_VS = 0;

    fann_reset_MSE(vnn->getNeuralNet());
    
    if(clear_trajectory){
      trajectory_a->clear();
      trajectory_v->clear();
    }
    
//     if(episode % 50 == 0)
    mean_sum_VS.clear();
    
    episode ++;
  }

  void end_episode() override {
    mean_sum_VS.push_back(sum_VS);
    
    if(!online_update){
      update_actor();
    
      removeOldPolicyTrajectory();
        
//         write_valuef_file("v_before.data");
      update_critic();
//         write_valuef_file("v_after.data");
     
    }
  }
  
  void write_valuef_file(const std::string& file){
      
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
        out << vnn->computeOutVF(x, {});
        out << std::endl;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 6);
      out.close();
/*
      function doit()
      close all; clear all; 
      X=load("v_after.data"); % X=load("v_before.data"); 
      tmp_ = X(:,2); X(:,2) = X(:,3); X(:,3) = tmp_;
      key = X(:, 1:2);
      for i=1:size(key, 1)
       subkey = find(sum(X(:, 1:2) == key(i,:), 2) == 2);
       data(end+1, :) = [key(i, :) mean(X(subkey, end))];
      endfor
      [xx,yy] = meshgrid (linspace (-1,1,300));
      griddata(data(:,1), data(:,2), data(:,3), xx, yy, "linear"); xlabel('theta_1'); ylabel('theta_2');
      endfunction
*/    
  }
  
  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<SASRG_sample>& _vtraj, const HybridCaclaAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        SASRG_sample sm = vtraj[n];
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
        SASRG_sample sm = vtraj[n];

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
    const std::vector<SASRG_sample>& vtraj;
    const HybridCaclaAg* ptr;
  };

  void removeOldPolicyTrajectory() {
    
    if(!update_vtraj_actor)
      return;

    for(auto iter = trajectory_v->tree().begin(); iter != trajectory_v->tree().end(); ) {
      SASRG_sample sm = *iter;
      vector<double> *ac = ann->computeOut(sm.s);

      double dist = bib::Utils::euclidien_dist(*ac, sm.a, 2);

      double proba = 1.f - pow(dist, transform_proba);
      if (!sm.goal_reached && proba > 0 && bib::Utils::rand01() < proba) {
        trajectory_v->tree().erase(iter++);
      } else {
        ++iter;
      }
      
      delete ac;
    }
  }


  void update_actor() {
    if(trajectory_a->size() > 0) {

      struct fann_train_data* data = fann_create_train(trajectory_a->size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory_a->tree().begin(); it != trajectory_a->tree().end() ; ++it) {
        SA_sample sm = *it;

        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = sm.a[i];

        n++;
      }

      ann->learn(data, 5000, 0, 0.0001);

      fann_destroy_train(data);
    }
  }

  void update_critic() {
    if (trajectory_v->size() > 0) {
      //remove trace of old policy

      std::vector<SASRG_sample> vtraj(trajectory_v->size());
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
    }
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
//     bib::XMLEngine::save<>(trajectory_v, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " <<
        std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " <<
        trajectory_v->size() << " " << trajectory_a->size() << " " << bib::Utils::statistics(mean_sum_VS).mean << " " << mean_sum_VS.size() << 
        " " << vnn->weightSum() << " " <<ann->weightSum();
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
  
  bool lecun_activation, online_update, update_pure_ac, kd_tree_autoremove, clear_trajectory, update_vtraj_actor,
  gaussian_policy;
  double transform_proba;
  
  std::vector<double> mean_sum_VS;
  double sum_VS;

  uint internal_time;
  uint decision_each;
  uint episode;

  double gamma, noise;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  Trajectory<SA_sample>* trajectory_a;
  Trajectory<SASRG_sample>* trajectory_v;

  MLP* ann;
  MLP* vnn;
};

#endif

