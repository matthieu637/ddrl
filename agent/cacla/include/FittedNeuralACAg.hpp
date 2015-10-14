#ifndef FITTEDNEURALACAG_HPP
#define FITTEDNEURALACAG_HPP

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

#define PRECISION 0.01
#define CONVERG_PRECISION 0.0001
#define DEBUG_FILE
#define BATCH_SIZE 200

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  double p0;
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


class FittedNeuralACAg : public arch::AAgent<> {
 public:
  FittedNeuralACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~FittedNeuralACAg() {    
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
      
      double p0 = 0.f;
      if(!gaussian_policy){
          if (std::equal(last_action->begin(), last_action->end(), last_pure_action->begin()))//no explo
            p0 = 1 - noise;
          else
            p0 = noise; //should be 0 but p0 > 0
      } else {
          p0 = 1.f;
          for(uint i=0;i < nb_motors;i++)
            p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      }

      trajectory_q_last.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0});
      proba_s.add_data(last_state);
      trajectory_a.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0});
      if(goal_reached)
        if(encourage_absorbing)
          trajectory_q_absorbing.push_back( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0});
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
    importance_sampling         = pt->get<bool>("agent.importance_sampling");
    is_multi_drawn              = pt->get<bool>("agent.is_multi_drawn");
    encourage_absorbing         = pt->get<bool>("agent.encourage_absorbing");
    importance_sampling_actor   = pt->get<bool>("agent.importance_sampling_actor");
    learnV                      = pt->get<bool>("agent.learnV");
    
    max_iteration = log(PRECISION) / log(gamma);
    LOG_DEBUG("gamma : " << gamma << " => max_iter : " << max_iteration );
        
    if(!gaussian_policy){
      noise = 0.15;
      LOG_DEBUG("greedy policy " << noise);
    }
    
    if(encourage_absorbing && learnV){
      LOG_ERROR("encourage_absorbing doesn't make sense when learning V");
      exit(1);
    }
    
    episode=-1;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);
    
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

    trajectory_q_last.clear();
    
    trajectory_a.clear();
    
    episode ++;
  }
  
  void update_actor(){
      if(trajectory_a.size() == 0)
        return;
    
      struct fann_train_data* data = fann_create_train(trajectory_a.size()+10, nb_sensors, nb_motors);
      
      uint n=0;
      KDE proba_s_explo;
      for(auto it = trajectory_a.begin(); it != trajectory_a.end() ; ++it) {
        sample sm = *it;
        
        if (std::equal(sm.a.begin(), sm.a.end(), sm.pure_a.begin()))
          continue;
        
        
        double exploration = sm.r;
        if (!sm.goal_reached){
            double nextV = vnn->computeOutVF(sm.next_s, {});
            exploration += gamma * nextV;
        } 
        double exploitation = vnn->computeOutVF(sm.s, {});
        
        write_policy_data("aP" + std::to_string(episode), sm.s, sm.a, exploration - exploitation);
                
        if(exploration > exploitation){
            proba_s_explo.add_data(sm.s);
            
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.a[i];
            n++;
        } 
      }
      
      bib::Logger::getInstance()->closeFile("aP" + std::to_string(episode));
      
      if(n == 0){//nothing new => quit
        fann_destroy_train(data);
        return;
      }
      

      MLP generative_model(*ann);
      
      std::vector<sample> vtraj;
      vtraj.reserve(trajectory_q.size() + trajectory_q_last.size());
      std::copy(trajectory_q.begin(), trajectory_q.end(), std::back_inserter(vtraj));
      std::copy(trajectory_q_last.begin(), trajectory_q_last.end(), std::back_inserter(vtraj));
      
//       proba_s.set_bandwidth_opt_type(2);
//       too long to compute
    
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        double p_s = 1 + proba_s.pdf(it->s);
        it->pfull_data = (1.f / (p_s));
      }
      
      auto eval = [&]() {
        return fann_get_MSE(ann->getNeuralNet());
      };
            
      auto iter = [&]() {
        uint n_local = n;
        std::vector<sample> vtraj_local;
        vtraj_local.reserve(vtraj.size());
        std::copy(vtraj.begin(), vtraj.end(), std::back_inserter(vtraj_local));
        
        std::vector<sample> vtraj_is_state;
        generateSubData(vtraj_is_state, vtraj_local, min(10, trajectory_q.size() + trajectory_q_last.size() ));
        
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
        
          vector<double>* next_action = generative_model.computeOut(sm.s);
          for (uint i = 0; i < nb_motors; i++)
            data->output[n_local][i] = next_action->at(i);
          delete next_action;
          
          n_local++;
        }
        
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n_local);

        ann->learn(subdata, 100, 0, CONVERG_PRECISION);

        fann_destroy_train(subdata);
      };

      if(importance_sampling_actor)
        bib::Converger::determinist<>(iter, eval, 10, CONVERG_PRECISION, 0);
      else
        iter();
//       for(uint i=0; i < 10 ; i++)
//         iter();

      fann_destroy_train(data);
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
      
      write_valuef_file("Q." + std::to_string(episode), vnn);

      write_valuef_added_file("aQ." + std::to_string(episode));
      
      write_policy_file("P." + std::to_string(episode));
      write_state_dis("pcs" + std::to_string(episode));
      
      update_critic();
      
      if(!co_update)
        update_actor();
      
      for (auto sm : trajectory_q_last)
        trajectory_q.insert(sm);
  }
  
  void write_state_dis(const std::string& file){
#ifdef DEBUG_FILE
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
#endif
  }
  
  void write_policy_addeddata(const std::string& file, vector< sample > vtraj_is, const MLP& actor){
#ifdef DEBUG_FILE
    std::ofstream out;
    out.open(file, std::ofstream::out | std::ofstream::app);
  
    for(auto sm : vtraj_is){
      out << sm.s[0] << " " ;

      vector<double> * ac = actor.computeOut(sm.s);
      out << ac->at(0);
      out << std::endl;
      delete ac;
    }
    
    out.close();
#endif
  }
  
  void write_valuef_added_file(const std::string& file){
#ifdef DEBUG_FILE
    std::ofstream out;
    out.open(file, std::ofstream::out | std::ofstream::app);
  
    for(auto sm : trajectory_q_absorbing){
      out << sm.s[0] << " " ;
      out << sm.r << " ";
//       out << sm.next_s[0] << " " ;
      out << std::endl;
    }
    
    out.close();
#endif
  }
  
  void write_valuef_point_file(const std::string& file, fann_train_data* data){
#ifdef DEBUG_FILE
     for(uint i = 0 ; i < data->num_data ; i++ ){
        LOG_FILE(file, data->input[i][0] << " " << data->output[i][0]);
     } 
     bib::Logger::getInstance()->closeFile(file);
#endif
  }
  
  void write_policy_data(const std::string& file, const std::vector<double>& s, 
                         const std::vector<double>& a, double explorMexploi){
#ifdef DEBUG_FILE
      LOG_FILE(file, s[0] << " " << a[0] << " " << explorMexploi);
#endif
  }
  
  void write_policy_file(const std::string& file){
#ifdef DEBUG_FILE
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
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 200);
      out.close();
#endif
  }
  
  void write_valuef_file(const std::string& file, const MLP* _vnn){
#ifdef DEBUG_FILE
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";

        out << _vnn->computeOutVF(x, {});
        out << std::endl;
      };
      
//       bib::Combinaison::continuous<>(iter, nb_sensors+nb_motors, -M_PI, M_PI, 6);
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 50);
      out.close();
#endif
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
    ParraVtoVNext(const std::vector<sample>& _vtraj, const FittedNeuralACAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {

      if(ptr->encourage_absorbing)
        data = fann_create_train(vtraj.size() + ptr->trajectory_q_absorbing.size(), ptr->nb_sensors, 1);
      else
        data = fann_create_train(vtraj.size(), ptr->nb_sensors, 1);
      
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
      }
      
      for (uint n = 0; n < ptr->trajectory_q_absorbing.size(); n++) {
        sample sm = ptr->trajectory_q_absorbing[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[vtraj.size() + n][i] = sm.s[i];
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
        sample sm = n < vtraj.size() ? vtraj[n] : ptr->trajectory_q_absorbing[n - vtraj.size()];

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
    const FittedNeuralACAg* ptr;
  };
  
  //importance sampling + multi drawn
  void createMiniBatchV(std::vector<sample>& vtraj_final, std::vector<sample>& vtraj_local){
      vtraj_final.clear();
      vtraj_final.reserve(BATCH_SIZE);
      
      if(vtraj_local.size() == 0){
        for(auto sm : trajectory_q_last){
            sm.pfull_data = 1;
            vtraj_local.push_back(sm);
        }
        
        for(auto sm : trajectory_q){
            std::vector<double>* next_action = ann->computeOut(sm.s);
            if(!gaussian_policy){
              if (std::equal(sm.a.begin(), sm.a.end(), next_action->begin()))
                sm.pfull_data =  (1 - noise) / sm.p0;
              else
                sm.pfull_data = 0;
//             sm.pfull_data = noise / sm.p0;
            } else {
              sm.pfull_data = 1.f;
              for(uint i=0;i < nb_motors;i++)
                sm.pfull_data *= exp(-(sm.a[i]-next_action->at(i))*(sm.a[i]-next_action->at(i))/(2.f*noise*noise));
            }
  
            delete next_action;
            
            vtraj_local.push_back(sm);
        }
      }
      
      std::vector<sample> vtraj_before_ps;
      generateSubData(vtraj_before_ps, vtraj_local, BATCH_SIZE);
      
      KDE proba_s_local;
      
      for(auto sm : vtraj_before_ps)
          proba_s_local.add_data(sm.s);
      
      proba_s.calc_bandwidth();
      
      for(auto sm : vtraj_before_ps)
          sm.pfull_data = 1 / ( 1 + proba_s_local.pdf( sm.s, proba_s.get_default_bandwidth_map(), 1.f ));
      
      generateSubData(vtraj_final, vtraj_before_ps, BATCH_SIZE);
  }

  //importance sampling
  void createBatchV(std::vector<sample>& vtraj_final, std::vector<sample>& vtraj_precompute){
      createMiniBatchV(vtraj_final, vtraj_precompute);
  }
  
  void update_critic() {
    if (trajectory_q.size() > 0 || trajectory_q_last.size() > 0) {
      //remove trace of old policy

      std::vector<sample> vtraj;
      std::vector<sample> vtraj_precompute;
      vtraj_precompute.reserve(trajectory_q.size() + trajectory_q_last.size());
      
      if(importance_sampling){
        if(is_multi_drawn)
          createMiniBatchV(vtraj, vtraj_precompute);
        else
          createBatchV(vtraj, vtraj_precompute);
      }else {
        vtraj.reserve(trajectory_q_last.size());
        std::copy(trajectory_q_last.begin(), trajectory_q_last.end(), std::back_inserter(vtraj));
      }
      //std::shuffle(vtraj.begin(), vtraj.end()); //useless due to rprop batch
      
      
      ParraVtoVNext* dq = nullptr; 
      if(!is_multi_drawn)
        dq = new ParraVtoVNext(vtraj, this);

      uint iteration = 0;
      auto iter = [&]() {
        if(is_multi_drawn){
          createMiniBatchV(vtraj, vtraj_precompute);
          dq = new ParraVtoVNext(vtraj, this);
        }
        
        if(encourage_absorbing)
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size() + trajectory_q_absorbing.size()), *dq);
        else
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), *dq);

        if(is_multi_drawn)
          write_valuef_file("V." + std::to_string(episode) + "." + std::to_string(iteration), vnn);
        
        if(vnn_from_scratch)
          fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
        
        if(is_multi_drawn){
          write_valuef_point_file("aaQ." + std::to_string(episode) + "." + std::to_string(iteration), dq->data);
          vnn->learn_stoch(dq->data, 100, 0, CONVERG_PRECISION);
        }
        else
          vnn->learn_stoch(dq->data, 10000, 0, CONVERG_PRECISION);
        
        if(co_update){
          update_actor();
          vtraj_precompute.clear();
        }
        
        if(is_multi_drawn){
          dq->free();
          delete dq;
        }
        iteration++;
      };

      auto eval = [&]() {
        return fann_get_MSE(vnn->getNeuralNet());
      };

      
      if(determinist_vnn_update)
         bib::Converger::determinist<>(iter, eval, max_iteration, CONVERG_PRECISION, 0);
      else {
        NN best_nn = nullptr;
        auto save_best = [&]() {
          if(best_nn != nullptr)
            fann_destroy(best_nn);
          best_nn = fann_copy(vnn->getNeuralNet());
        };

        bib::Converger::min_stochastic<>(iter, eval, save_best, max_iteration, CONVERG_PRECISION, 0, 10);
        vnn->copy(best_nn);
        fann_destroy(best_nn);
      }

      if(!is_multi_drawn){
        dq->free();
        delete dq;
      }
      
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
        trajectory_q.size() << " " << vnn->weightSum() << " " << trajectory_q_absorbing.size() << " "
        << trajectory_q_last.size();
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
  uint max_iteration;
  bool lecun_activation, determinist_vnn_update, gaussian_policy,
    clear_trajectory, vnn_from_scratch, co_update, 
    importance_sampling, is_multi_drawn, encourage_absorbing,
    importance_sampling_actor, learnV;
  
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

  KDE proba_s;
  KDE proba_s_ct;
  std::set<sample> trajectory_q;
  std::set<sample> trajectory_q_last;
  std::set<sample> trajectory_a;
  std::vector<sample> trajectory_q_absorbing;

  MLP* vnn;
  MLP* ann;
};

#endif

