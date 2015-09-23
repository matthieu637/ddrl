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

class FittedQACAg : public arch::AAgent<> {
 public:
  FittedQACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~FittedQACAg() {
    delete trajectory;
    
    delete vnn;
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

      QSASRG_sample vsampl = {last_state, *last_action, sensors, reward, goal_reached};
      trajectory->addPoint(vsampl); 
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      //gaussian policy
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      }else {
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
    gamma                       = pt->get<double>("agent.gamma");
    hidden_unit_v               = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a               = pt->get<int>("agent.hidden_unit_a");
    noise                       = pt->get<double>("agent.noise");
    decision_each               = pt->get<int>("agent.decision_each");
    lecun_activation            = pt->get<bool>("agent.lecun_activation");
    kd_tree_autoremove          = pt->get<bool>("agent.kd_tree_autoremove");
    transform_proba             = pt->get<double>("agent.transform_proba");
    determinist_vnn_update      = pt->get<bool>("agent.determinist_vnn_update");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    clear_trajectory            = pt->get<bool>("agent.clear_trajectory");
    vnn_from_scratch            = pt->get<bool>("agent.vnn_from_scratch");

    if(!gaussian_policy)
      noise = 0.05;
    
    episode=-1;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);
    
    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);

    //don't need normalization if removeTrajectory
    trajectory = new Trajectory<QSASRG_sample>(nb_sensors, transform_proba, kd_tree_autoremove);
    
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
    
    
    sum_VS = 0;

    fann_reset_MSE(vnn->getNeuralNet());
    
    if(clear_trajectory)
      trajectory->clear();
    
    mean_sum_VS.clear();
    
    episode ++;
  }
  
  void update_actor(){
    if (trajectory->size() > 0) {

      struct fann_train_data* data = fann_create_train(trajectory->size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory->tree().begin(); it != trajectory->tree().end() ; ++it) {
          QSASRG_sample sm = *it;

          vector<double>* ac = vnn->optimized(sm.s, {}, 2);
        
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          for (uint i = 0; i < nb_motors; i++)
            data->output[n][i] = ac->at(i);
          
          delete ac;
      }
      
      ann->learn(data, 5000, 0, 0.0001);

      fann_destroy_train(data);
    }
  }

  void end_episode() override {
    mean_sum_VS.push_back(sum_VS);
           
//         write_valuef_file("v_before.data");
      update_critic();
//         write_valuef_file("v_after.data");

//     write_valuef_file("v_after.data." + std::to_string(episode));
  }
  
  void write_valuef_file(const std::string& file){
      
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
          
        std::vector<double> m(nb_motors);
        for(uint i=nb_sensors; i < nb_motors + nb_sensors;i++)
          m[i - nb_sensors] = x[nb_sensors + i];
        out << vnn->computeOut(x, m);
        out << std::endl;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors+nb_motors, -M_PI, M_PI, 6);
      out.close();
/*
      close all; clear all; 
      function doit(ep)
        X=load(strcat("v_after.data.",num2str(ep)));
        tmp_ = X(:,1); X(:,1) = X(:,3); X(:,3) = tmp_;
        tmp_ = X(:,2); X(:,2) = X(:,5); X(:,5) = tmp_;
        key = X(:, 1:2);
        for i=1:size(key, 1)
        subkey = find(sum(X(:, 1:2) == key(i,:), 2) == 2);
        data(end+1, :) = [key(i, :) mean(X(subkey, end))];
        endfor
        [xx,yy] = meshgrid (linspace (-pi,pi,300));
        griddata(data(:,1), data(:,2), data(:,3), xx, yy, "linear"); xlabel('theta'); ylabel('a');
      endfunction
*/    
  }
  
  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<QSASRG_sample>& _vtraj, const FittedQACAg* _ptr) : vtraj(_vtraj), ptr(_ptr),
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
      
      for(auto it : actions)
        delete it;
    }
    
    void update_actions(){
      for(auto it : actions)
        delete it;
      
      for (uint n = 0; n < vtraj.size(); n++){
        QSASRG_sample sm = vtraj[n];
        actions[n] = ptr->ann->computeOut(sm.next_s);
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
    const FittedQACAg* ptr;
    std::vector<std::vector<double>*> actions;
  };

  void update_critic() {
    if (trajectory->size() > 0) {
      //remove trace of old policy

      std::vector<QSASRG_sample> vtraj(trajectory->size());
      std::copy(trajectory->tree().begin(), trajectory->tree().end(), vtraj.begin());
      //std::shuffle(vtraj.begin(), vtraj.end()); //useless due to rprop batch

      ParraVtoVNext dq(vtraj, this);

      auto iter = [&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);

        if(vnn_from_scratch)
          fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
        vnn->learn(dq.data, 10000, 0, 0.0001);
        
        update_actor();
        dq.update_actions();
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
      //       LOG_DEBUG("number of data " << trajectory.size());
    }
  }

  void save(const std::string& path) override {
    vnn->save(path+".critic");
//     bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    vnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " <<
        std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " <<
        trajectory->size() << " " << bib::Utils::statistics(mean_sum_VS).mean << " " << mean_sum_VS.size() << 
        " " << vnn->weightSum();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory->size() ;
  }

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;
  
  double alpha_a;
  bool lecun_activation, kd_tree_autoremove, determinist_vnn_update, gaussian_policy,
    clear_trajectory, vnn_from_scratch;
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

  Trajectory<QSASRG_sample>* trajectory;

  MLP* vnn;
  MLP* ann;
};

#endif

