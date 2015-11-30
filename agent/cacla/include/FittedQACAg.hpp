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
#include "../../qlearning-nn/include/MLP.hpp"
#include "kde.hpp"

#define BATCH_SIZE 200

// #define ALTERN_ACTOR_UPDATE_HYBRID_SAMPLE_ARGMAX

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  double pfull_data;

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


class FittedQACAg : public arch::AAgent<> {
 public:
  FittedQACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~FittedQACAg() {    
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
  
  void mergeSA(std::vector<double>& AB, const std::vector<double>& A, const std::vector<double>& B) {
    AB.reserve( A.size() + B.size() ); // preallocate memory
    AB.insert( AB.end(), A.begin(), A.end() );
    AB.insert( AB.end(), B.begin(), B.end() );
  }

  const std::vector<double>& _runf(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) {

    
    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {  // Update Q

      trajectory_q.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, 0});
      std::vector<double> sa;
      mergeSA(sa, last_state, *last_action);
      proba_sa.add_data(sa);
      proba_s.add_data(last_state);
      trajectory_a.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, 0});
      if(goal_reached)
        trajectory_q_absorbing.push_back( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, 0});
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
    determinist_vnn_update      = pt->get<bool>("agent.determinist_vnn_update");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    clear_trajectory            = pt->get<bool>("agent.clear_trajectory");
    vnn_from_scratch            = pt->get<bool>("agent.vnn_from_scratch");
    co_update                   = pt->get<bool>("agent.co_update");
    update_pure_ac              = pt->get<bool>("agent.update_pure_ac");
    sample_update               = pt->get<bool>("agent.sample_update");
    importance_sampling         = pt->get<bool>("agent.importance_sampling");
    is_multi_drawn              = pt->get<bool>("agent.is_multi_drawn");
    alternative_actor_update    = pt->get<bool>("agent.alternative_actor_update");
    actor_update_argmax         = pt->get<bool>("agent.actor_update_argmax");
    encourage_absordbing        = pt->get<bool>("agent.encourage_absordbing");
    importance_sampling_actor   = pt->get<bool>("agent.importance_sampling_actor");

    if(!gaussian_policy){
      noise = 0.15;
      LOG_DEBUG("greedy policy " << noise);
    }
    
    episode=-1;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);
    
    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
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
  

    fann_reset_MSE(vnn->getNeuralNet());
    
    if(clear_trajectory)
      trajectory_q.clear();
    
    trajectory_a.clear();
    
    episode ++;
  }
  
  void altern_actor_update(){
      if(trajectory_a.size() == 0)
        return;
    
      struct fann_train_data* data = fann_create_train(trajectory_a.size()+10, nb_sensors, nb_motors);
      
      uint n=0;
      KDE proba_s_explo;
      for(auto it = trajectory_a.begin(); it != trajectory_a.end() ; ++it) {
        sample sm = *it;
        
        if (std::equal(sm.a.begin(), sm.a.end(), sm.pure_a.begin()))
          continue;
        
        double exploration = vnn->computeOut(sm.next_s, sm.a);
        double exploitation = vnn->computeOut(sm.next_s, sm.pure_a);
                
        if(exploration > exploitation){
            proba_s_explo.add_data(sm.s);
            
            
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.a[i];
            n++;
        } 
      }
      
      if(n == 0)//nothing new => quit
        return;
      

      MLP generative_model(*ann);
      
      std::vector<sample> vtraj(trajectory_q.size());
      std::copy(trajectory_q.begin(), trajectory_q.end(), vtraj.begin());
      
//       proba_s.set_bandwidth_opt_type(2);
//       too long to compute
    
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        double p_s = 1 + proba_s.pdf(it->s);
        it->pfull_data = (1.f / (p_s));
      }
      
//       auto eval = [&]() {
//         return fann_get_MSE(ann->getNeuralNet());
//       };
            
      auto iter = [&]() {
        uint n_local = n;
        std::vector<sample> vtraj_local(vtraj.size());
        std::copy(vtraj.begin(), vtraj.end(), vtraj_local.begin());
        
        std::vector<sample> vtraj_is_state;
        generateSubData(vtraj_is_state, vtraj_local, min(10, trajectory_q.size()));
        
        proba_s.calc_bandwidth();
        for(auto sm : vtraj_is_state) {
          double p_s_expl = 1 + proba_s_explo.pdf(sm.s, proba_s.get_default_bandwidth_map(), 1.5f);
          p_s_expl = p_s_expl * p_s_expl;
          sm.pfull_data = (1.f / (p_s_expl)) ;
        }
        
        std::vector<sample> vtraj_is;
        generateSubData(vtraj_is, vtraj_is_state, vtraj_is_state.size());
        write_policy_addeddata("aaP" + std::to_string(episode), vtraj_is, generative_model);
         
        for(sample sm : vtraj_is) {
          for (uint i = 0; i < nb_sensors ; i++)
              data->input[n_local][i] = sm.s[i];
        
#ifdef ALTERN_ACTOR_UPDATE_HYBRID_SAMPLE_ARGMAX
//           vector<double>* next_action = vnn->optimized(sm.s, {}, 4);
          vector<double>* next_action = vnn->optimizedBruteForce(sm.s, 0.05);
#else
          vector<double>* next_action = generative_model.computeOut(sm.s);
#endif
          for (uint i = 0; i < nb_motors; i++)
            data->output[n_local][i] = next_action->at(i);
          delete next_action;
          
          n_local++;
        }
        
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n_local);

//         ann->learn_stoch(subdata, 5000, 0, 0.0001);
        ann->learn(subdata, 100, 0, 0.00001);

        fann_destroy_train(subdata);
      };

//       bib::Converger::determinist<>(iter, eval, 10, 0.0001, 0);
      for(uint i=0; i < 10 ; i++)
        iter();

      fann_destroy_train(data);
  }
  
  void altern_actor_update_argmax(){
    if (trajectory_q.size() > 0) {

      if(!importance_sampling_actor){
        struct fann_train_data* data = fann_create_train(trajectory_q.size(), nb_sensors, nb_motors);

        uint n=0;
        for(auto it = trajectory_q.begin(); it != trajectory_q.end() ; ++it) {
            sample sm = *it;

            vector<double>* ac = vnn->optimizedBruteForce(sm.s, 0.05f);
  //           vector<double>* ac = vnn->optimized(sm.s, {}, 4);
          
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = ac->at(i);
            
            delete ac;
        }
        
        ann->learn(data, 1000, 0, 0.0001);

        fann_destroy_train(data);
      } else {
        
        
      }
    }
  }
  
  void update_actor(){
    if(alternative_actor_update){
      if(actor_update_argmax)
        altern_actor_update_argmax();
      else
        altern_actor_update();
        return;
    }
    
    
    
    if (trajectory_a.size() > 0) {

      struct fann_train_data* data = fann_create_train(trajectory_a.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory_a.begin(); it != trajectory_a.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        if(sample_update){
          target = vnn->computeOut(sm.next_s, sm.a);
          mine = vnn->computeOut(sm.next_s, sm.pure_a);
        } else {
          target = sm.r;
          if (!sm.goal_reached) {
            vector<double>* next_action = ann->computeOut(sm.next_s);
            double nextV = vnn->computeOut(sm.next_s, *next_action);
            target += gamma * nextV;
            delete next_action;
          }
          mine = vnn->computeOut(sm.s, sm.a);
        }
      
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
        }
      }

      if(n > 0) {
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);

        ann->learn(subdata, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
    }
  }

  void generateSubData(vector< sample >& vtraj_is, const vector< sample >& vtraj_full, uint length){
      std::list<double> weights;
      
      for(auto it = vtraj_full.cbegin(); it != vtraj_full.cend() ; ++it){
          ASSERT(!std::isnan(it->pfull_data), "nan proba");
          weights.push_back(it->pfull_data);
      }
      
      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      std::set<int> uniq_index;
      for(uint i = 0; i < length; i++)
        uniq_index.insert(dist(*bib::Seed::random_engine()));
      
      for(auto i : uniq_index)
        vtraj_is.push_back(vtraj_full[i]);
  }
  
  void end_episode() override {
      
      write_valuef_file("Q." + std::to_string(episode));
      write_policy_file("P." + std::to_string(episode));
      write_state_dis("pcs" + std::to_string(episode));
           
      update_critic();
      
      write_policy_data("aP" + std::to_string(episode));

      if(!co_update)
        update_actor();
  }
  
  void write_state_dis(const std::string& file){
      std::ofstream out;
      out.open(file, std::ofstream::out);
      
      if(trajectory_a.size() > 0){
        KDE kde;
        for(auto sm : trajectory_a){
          kde.add_data(sm.s);
        }
        
        auto iter = [&](std::vector<double>& x) {
          for(uint i=0;i < x.size();i++)
            out << x[i] << " ";
          
          out << kde.pdf(x);
          out << std::endl;
        };
        
        bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 50);
      }
      out.close();
  }
  
  void write_policy_addeddata(const std::string& file, vector< sample > vtraj_is, const MLP& actor){
    std::ofstream out;
    out.open(file, std::ofstream::out | std::ofstream::app);
  
    for(auto sm : vtraj_is){
      out << sm.s[0] << " " ;

#ifdef ALTERN_ACTOR_UPDATE_HYBRID_SAMPLE_ARGMAX
//       vector<double>* ac = vnn->optimized(sm.s, {}, 4);
      vector<double>* ac = vnn->optimizedBruteForce(sm.s, 0.05);
#else
      vector<double> * ac = actor.computeOut(sm.s);
#endif
      out << ac->at(0);
      out << std::endl;
      delete ac;
    }
    
    out.close();
  }
  
  void write_policy_data(const std::string& file){
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      for(auto sm : trajectory_a){
        out << sm.s[0] << " " ;
        out << sm.a[0] << " " ;
        double explor = vnn->computeOut(sm.s, sm.a);
        double exploi = vnn->computeOut(sm.s, sm.pure_a);
        out << (explor - exploi) << " " ;
        out << std::endl;
      }
      
      out.close();
  }
  
  void write_policy_file(const std::string& file){
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
        
        vector<double> * ac = ann->computeOut(x);
        out << ac->at(0);
        out << std::endl;
        delete ac;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 50);
      out.close();
  }
  
  void write_valuef_file(const std::string& file){
      
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
          
        std::vector<double> m(nb_motors);
        for(uint i=nb_sensors; i < nb_motors + nb_sensors;i++)
          m[i - nb_sensors] = x[i];
        out << vnn->computeOut(x, m);
        out << std::endl;
      };
      
//       bib::Combinaison::continuous<>(iter, nb_sensors+nb_motors, -M_PI, M_PI, 6);
      bib::Combinaison::continuous<>(iter, nb_sensors+nb_motors, -1, 1, 50);
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
    ParraVtoVNext(const std::vector<sample>& _vtraj, const FittedQACAg* _ptr) : vtraj(_vtraj), ptr(_ptr),
      actions(vtraj.size()) {

      if(ptr->encourage_absordbing)
        data = fann_create_train(vtraj.size() + ptr->trajectory_q_absorbing.size(), ptr->nb_sensors + ptr->nb_motors, 1);
      else
        data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors, 1);
      
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];

        actions[n] = ptr->ann->computeOut(sm.next_s);
      }
      
      for (uint n = 0; n < ptr->trajectory_q_absorbing.size(); n++) {
        sample sm = ptr->trajectory_q_absorbing[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[vtraj.size() + n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[vtraj.size() + n][i] = sm.a[i - ptr->nb_sensors];
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
        sample sm = vtraj[n];
        actions[n] = ptr->ann->computeOut(sm.next_s);
      }
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = n < vtraj.size() ? vtraj[n] : ptr->trajectory_q_absorbing[n - vtraj.size()];

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
    const std::vector<sample>& vtraj;
    const FittedQACAg* ptr;
    std::vector<std::vector<double>*> actions;
  };

  void update_critic() {
    if (trajectory_q.size() > 0) {
      //remove trace of old policy

      std::vector<sample> vtraj(trajectory_q.size());
      std::copy(trajectory_q.begin(), trajectory_q.end(), vtraj.begin());
      //std::shuffle(vtraj.begin(), vtraj.end()); //useless due to rprop batch
      
      std::vector<sample>* vtraj_final = &vtraj;
      if(importance_sampling){
      //compute pfull_data
        for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
              std::vector<double> sa;
              mergeSA(sa, it->s, it->a);
              it->pfull_data = (1.f / proba_sa.pdf(sa));
  //             it->pfull_data = 1.f / (it->ptheta_data) ;
  //             it->pfull_data = (1.f / proba_s.pdf(it->s));
  //             it->pfull_data = (1.f / proba_s.pdf(it->s)) * it->ptheta_data;
        }
        
        if(!is_multi_drawn){
          vtraj_final = new std::vector<sample>;
          generateSubData(*vtraj_final, vtraj, BATCH_SIZE);
        }
      }

      ParraVtoVNext* dq = nullptr; 
      if(!is_multi_drawn)
        dq = new ParraVtoVNext(*vtraj_final, this);

      auto iter = [&]() {
        if(is_multi_drawn){
          vtraj_final = new std::vector<sample>;
          generateSubData(*vtraj_final, vtraj, BATCH_SIZE);
          dq = new ParraVtoVNext(*vtraj_final, this);
        }
        if(encourage_absordbing)
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_final->size() + trajectory_q_absorbing.size()), *dq);
        else
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_final->size()), *dq);

        if(vnn_from_scratch)
          fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
        
//         LOG_DEBUG("longer step begin ...");
        if(is_multi_drawn)
          vnn->learn_stoch(dq->data, 100, 0, 0.0001);
        else
          vnn->learn_stoch(dq->data, 10000, 0, 0.0001);
//         LOG_DEBUG("longer step end ...");
        
        if(co_update){
          update_actor();
          if(!is_multi_drawn)
            dq->update_actions();
        }
        
        if(is_multi_drawn){
          dq->free();
          delete dq;
        }
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

      if(!is_multi_drawn){
        dq->free();
        delete dq;
      }
      
      if(importance_sampling)
        delete vtraj_final;
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
        trajectory_q.size() << " " << vnn->weightSum() << " " << trajectory_q_absorbing.size();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory_q.size() ;
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
  bool lecun_activation, determinist_vnn_update, gaussian_policy,
    clear_trajectory, vnn_from_scratch, co_update, update_pure_ac, sample_update, 
    importance_sampling, is_multi_drawn, alternative_actor_update, encourage_absordbing,
    actor_update_argmax, importance_sampling_actor;
  
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

  KDE proba_sa, proba_s;
  std::set<sample> trajectory_q;
  std::set<sample> trajectory_a;
  std::vector<sample> trajectory_q_absorbing;

  MLP* vnn;
  MLP* ann;
};

#endif

