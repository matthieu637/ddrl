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
#include "Critic.hpp"
#include "../../qlearning-nn/include/MLP.hpp"
#include "kde.hpp"

#define PRECISION 0.01
#define CONVERG_PRECISION 0.00001

// #ifndef NDEBUG
//  #define DEBUG_FILE
// #endif
// #define DEBUG_FILE

#define POINT_FOR_ONE_DIMENSION 30
#define FACTOR_DIMENSION_INCREASE 0.2

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

inline void mergeSA(std::vector<double>& AB, const std::vector<double>& A, const std::vector<double>& B) {
  AB.reserve( A.size() + B.size() ); // preallocate memory
  AB.insert( AB.end(), A.begin(), A.end() );
  AB.insert( AB.end(), B.begin(), B.end() );
}

class FittedNeuralACAg : public arch::AAgent<> {
 public:
  FittedNeuralACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~FittedNeuralACAg() {  
    delete critic;
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
      if(regularize_p0_distribution){
        if(!gaussian_policy){
            if (std::equal(last_action->begin(), last_action->end(), last_pure_action->begin()))//no explo
              p0 = 1 - noise;
            else
              p0 = noise * 0.5f; //should be 0 but p0 > 0
        } else {
            p0 = 1.f;
            for(uint i=0;i < nb_motors;i++)
              p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
        }
      }

      sample sm = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0};
      trajectory_q_last.insert(sm);
      proba_s.add_data(last_state);
      
      std::vector<double> sa;
      mergeSA(sa, last_state, *last_action);
      proba_sa.add_data(sa);
      
      trajectory_a.insert(sm);
      
      if(online_actor_update)
        update_actor_online(sm);
      
      if(goal_reached)
        if(encourage_absorbing)
          trajectory_q_absorbing.push_back(sm);
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
    gamma                               = pt->get<double>("agent.gamma");
    hidden_unit_v                       = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a                       = pt->get<int>("agent.hidden_unit_a");
    noise                               = pt->get<double>("agent.noise");
    decision_each                       = pt->get<int>("agent.decision_each");
    
    vnn_from_scratch                    = pt->get<bool>("agent.vnn_from_scratch");
    online_actor_update                 = pt->get<bool>("agent.online_actor_update");
    
    lecun_activation                    = pt->get<bool>("agent.lecun_activation");
    gaussian_policy                     = pt->get<bool>("agent.gaussian_policy");
    clear_trajectory                    = pt->get<bool>("agent.clear_trajectory");
    importance_sampling                 = pt->get<bool>("agent.importance_sampling");
    is_multi_drawn                      = pt->get<bool>("agent.is_multi_drawn");
    encourage_absorbing                 = pt->get<bool>("agent.encourage_absorbing");
    importance_sampling_actor           = pt->get<bool>("agent.importance_sampling_actor");
    learnV                              = pt->get<bool>("agent.learnV");
    regularize_space_distribution       = pt->get<bool>("agent.regularize_space_distribution");
    regularize_p0_distribution          = pt->get<bool>("agent.regularize_p0_distribution");
    regularize_pol_distribution         = pt->get<bool>("agent.regularize_pol_distribution");
    sample_update                       = pt->get<bool>("agent.sample_update");
    td_sample_actor_update              = pt->get<bool>("agent.td_sample_actor_update");
    actor_update_determinist_range      = pt->get<bool>("agent.actor_update_determinist_range");
    
    max_iteration = log(PRECISION) / log(gamma);
    LOG_DEBUG("gamma : " << gamma << " => max_iter : " << max_iteration );
    
    max_actor_batch_size = POINT_FOR_ONE_DIMENSION;
    double local_increment = 0;
    for(uint i = 1; i <= nb_sensors; i++)
      local_increment += 1.f / pow(i, FACTOR_DIMENSION_INCREASE);
    max_actor_batch_size *= local_increment;
    
    if(learnV)
      max_critic_batch_size = max_actor_batch_size;
    else {
      for(uint i= 1 + nb_sensors; i <= nb_sensors + nb_motors ; i++)
        local_increment += 1.f / pow(i, FACTOR_DIMENSION_INCREASE);
      max_critic_batch_size = POINT_FOR_ONE_DIMENSION * local_increment;
    }
    
    LOG_DEBUG("max_actor_batch_size : " << max_actor_batch_size << " - max_critic_batch_size : " << max_critic_batch_size);
      
    if(!gaussian_policy){
      noise = 0.1;
      LOG_DEBUG("greedy policy " << noise);
    }
    
    if(encourage_absorbing && learnV){
      LOG_ERROR("encourage_absorbing doesn't make sense when learning V");
      exit(1);
    }
    
    if(td_sample_actor_update && learnV){
      LOG_ERROR("td_sample_actor_update doesn't make sense when learning V");
      exit(1);
    }
    
    if(regularize_p0_distribution && !importance_sampling){
      LOG_ERROR("regularize_p0_distribution doesn't make sense without importance_sampling");
      exit(1);
    }
    
    if(regularize_pol_distribution && !importance_sampling){
      LOG_ERROR("regularize_pol_distribution doesn't make sense without importance_sampling");
      exit(1);
    }
    
    if(regularize_space_distribution && !importance_sampling){
      LOG_ERROR("regularize_space_distribution doesn't make sense without importance_sampling");
      exit(1);
    }
    
    if(td_sample_actor_update && sample_update){
      LOG_ERROR("td_sample_actor_update implies not sample_update");
      exit(1);
    }
    
    if(sample_update && learnV){
      LOG_ERROR("sample_update doesn't make sense when learning V");
      exit(1);
    }
    
    if(!importance_sampling && learnV && !clear_trajectory){
      LOG_ERROR("use all the data when learning V without clearing last trajectory is a non sense");
      exit(1);
    }
    
    episode=-1;
    
    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
    
    critic = new Critic<sample>(nb_sensors, nb_motors, learnV, hidden_unit_v, lecun_activation, ann, gamma);
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
     
    if(clear_trajectory)
      trajectory_q.clear();

    trajectory_q_last.clear();
    
    trajectory_a.clear();
    
    episode ++;
  }
  
  void update_actor_online(const sample& sm){
      struct fann_train_data* data = fann_create_train(max_actor_batch_size+2, nb_sensors, nb_motors);

      uint n=0;
      KDE proba_s_explo;
      
      if (std::equal(sm.a.begin(), sm.a.end(), sm.pure_a.begin())){
        fann_destroy_train(data);
        return;
      }
      
      double exploration, exploitation;
      if(!td_sample_actor_update){
        if(!sample_update){//TD error
          exploration = critic->evaluateExploration(sm);
          exploitation = critic->evaluateExploitation(sm);
        } else { // Sample error
          exploration = critic->getMLP()->computeOutVF(sm.next_s, sm.a);
          exploitation = critic->getMLP()->computeOutVF(sm.next_s, sm.pure_a);
        }
      } else {
        exploration = critic->evaluateExploration(sm);
        exploitation = critic->evaluateExploitation(sm);
        if(exploration < exploitation) {
          exploration = critic->getMLP()->computeOutVF(sm.next_s, sm.a);
          exploitation = critic->getMLP()->computeOutVF(sm.next_s, sm.pure_a); 
        }
      }
      
      write_policy_data("aP" + std::to_string(episode), sm.s, sm.a, exploration - exploitation);
      
      if(exploration > exploitation){
        proba_s_explo.add_data(sm.s);
        
        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = sm.a[i];
        n++;
      }
      
      bib::Logger::getInstance()->closeFile("aP" + std::to_string(episode));
      
      if(n == 0 || trajectory_q.size() + trajectory_q_last.size() < 2 ){//nothing new => quit
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
        generateSubData(vtraj_is_state, vtraj_local, min(max_actor_batch_size, trajectory_q.size() + trajectory_q_last.size() ));
        
        proba_s.calc_bandwidth();
        std::vector<sample> vtraj_is;
        
        if(!actor_update_determinist_range){
          for(auto _sm : vtraj_is_state) {
            double p_s_expl = 1 + proba_s_explo.pdf(_sm.s, proba_s.get_default_bandwidth_map(), 1.5f);
            p_s_expl = p_s_expl * p_s_expl;
            _sm.pfull_data = (1.f / (p_s_expl)) ;
          }
          
          generateSubData(vtraj_is, vtraj_is_state, vtraj_is_state.size());
        } else {
          for(auto _sm : vtraj_is_state){
            bool data_ok = true;
            for(uint i=0; i < nb_sensors && data_ok; i++)
              if(fabs(_sm.s[i] - sm.s[i]) < proba_s.get_default_bandwidth_map().at(i)){
                data_ok = false;
                break;
              }
            if(data_ok)
              vtraj_is.push_back(_sm);
          }
        }
        
        write_policy_addeddata("aaP" + std::to_string(episode), vtraj_is, generative_model);
         
        for(sample _sm : vtraj_is) {
          for (uint i = 0; i < nb_sensors ; i++)
              data->input[n_local][i] = _sm.s[i];
        
          vector<double>* next_action = generative_model.computeOut(_sm.s);
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
  
  void update_actor(){
      if(trajectory_a.size() == 0)
        return;
    
      struct fann_train_data* data = fann_create_train(trajectory_a.size()+max_actor_batch_size, nb_sensors, nb_motors);
      
      uint n=0;
      KDE proba_s_explo;
      std::vector<sample> vtraj_good_exploration;
      for(auto it = trajectory_a.begin(); it != trajectory_a.end() ; ++it) {
        sample sm = *it;
        
        if (std::equal(sm.a.begin(), sm.a.end(), sm.pure_a.begin()))
          continue;
        
        double exploration, exploitation;
        if(!td_sample_actor_update){
          if(!sample_update){//TD error
            exploration = critic->evaluateExploration(sm);
            exploitation = critic->evaluateExploitation(sm);
          } else { // Sample error
            exploration = critic->getMLP()->computeOutVF(sm.next_s, sm.a);
            exploitation = critic->getMLP()->computeOutVF(sm.next_s, sm.pure_a);
          }
        } else {
          exploration = critic->evaluateExploration(sm);
          exploitation = critic->evaluateExploitation(sm);
          if(exploration < exploitation) {
            exploration = critic->getMLP()->computeOutVF(sm.next_s, sm.a);
            exploitation = critic->getMLP()->computeOutVF(sm.next_s, sm.pure_a); 
          }
        }
        
        write_policy_data("aP" + std::to_string(episode), sm.s, sm.a, exploration - exploitation);
                
        if(exploration > exploitation){
            proba_s_explo.add_data(sm.s);
            
            vtraj_good_exploration.push_back(sm);
            
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.a[i];
            n++;
        }
      }
      
      bib::Logger::getInstance()->closeFile("aP" + std::to_string(episode));
      
      if(n == 0 || trajectory_q.size() + trajectory_q_last.size() < 2 ){//nothing new => quit
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
//       proba_s.calc_bandwidth();
//       LOG_DEBUG(proba_s.get_default_bandwidth_map().at(0));
      
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
        generateSubData(vtraj_is_state, vtraj_local, min(max_actor_batch_size, trajectory_q.size() + trajectory_q_last.size() ));
        
        // alternative
        // si un des points est trop proche (< bandwidth) à ceux à apprendre => retirer
        
        proba_s.calc_bandwidth();
        std::vector<sample> vtraj_is;
        
        if(!actor_update_determinist_range){
          for(auto sm : vtraj_is_state) {
            double p_s_expl = 1 + proba_s_explo.pdf(sm.s, proba_s.get_default_bandwidth_map(), 1.5f);
            p_s_expl = p_s_expl * p_s_expl;
            sm.pfull_data = (1.f / (p_s_expl)) ;
          }
          generateSubData(vtraj_is, vtraj_is_state, vtraj_is_state.size());
        } else {
          for(auto sm : vtraj_is_state){
            bool data_ok = true;
            for(uint i=0; i < nb_sensors && data_ok; i++)
              for(auto sm2 : vtraj_good_exploration)
              if(fabs(sm.s[i] - sm2.s[i]) < proba_s.get_default_bandwidth_map().at(i)){
                data_ok = false;
                break;
              }
            if(data_ok)
              vtraj_is.push_back(sm);
          }
        }
        
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
      critic->write_critic_file(std::to_string(episode));

      write_valuef_added_file("aQ." + std::to_string(episode));
      
//       if(episode % 10 == 0)
//         write_policy_file("P." + std::to_string(episode));
      write_state_dis("pcs" + std::to_string(episode));
      
      if(!online_actor_update){
        if(episode % 10 == 0)
          update_actor(); 
      }
      
      update_critic();
      
      for (auto sm : trajectory_q_last)
        trajectory_q.insert(sm);
  }

#ifdef DEBUG_FILE
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
#else
  void write_state_dis(const std::string&){
#endif
  }
  
#ifdef DEBUG_FILE
  void write_policy_addeddata(const std::string& file, vector< sample > vtraj_is, const MLP& actor){

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
#else
  void write_policy_addeddata(const std::string&, vector< sample >, const MLP&){
#endif
  }
  
#ifdef DEBUG_FILE
  void write_valuef_added_file(const std::string& file){

    std::ofstream out;
    out.open(file, std::ofstream::out | std::ofstream::app);
  
    for(auto sm : trajectory_q_absorbing){
      out << sm.s[0] << " " ;
      if(!learnV)
        out << sm.a[0] << " " ;
      out << sm.r << " ";
//       out << sm.next_s[0] << " " ;
      out << std::endl;
    }
    
    out.close();
#else
  void write_valuef_added_file(const std::string&){
#endif
  }
  
#ifdef DEBUG_FILE
  void write_valuef_point_file(const std::string& file, fann_train_data* data){

     for(uint i = 0 ; i < data->num_data ; i++ ){
        LOG_FILE(file, data->input[i][0] << " " << data->output[i][0]);
     } 
     bib::Logger::getInstance()->closeFile(file);
#else
  void write_valuef_point_file(const std::string&, fann_train_data*){
#endif
  }

#ifdef DEBUG_FILE
  void write_policy_data(const std::string& file, const std::vector<double>& s, 
                         const std::vector<double>& a, double explorMexploi){
      LOG_FILE(file, s[0] << " " << a[0] << " " << explorMexploi);
#else
  void write_policy_data(const std::string& , const std::vector<double>& , 
                         const std::vector<double>&, double){
#endif
  }
  
  void write_policy_file(const std::string& file){
// #ifdef DEBUG_FILE
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
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 100);
      out.close();
// #endif
  }
  
  //importance sampling + multi drawn
  void createMiniBatchV(std::vector<sample>& vtraj_final, std::vector<sample>& vtraj_local){
      vtraj_final.clear();
      vtraj_final.reserve(max_critic_batch_size);
      
      if(vtraj_local.size() == 0){
        for(auto sm : trajectory_q_last){
            sm.pfull_data = 1;
            vtraj_local.push_back(sm);
        }
        
        for(auto sm : trajectory_q){
            std::vector<double>* next_action = ann->computeOut(sm.s);
            if(regularize_pol_distribution){
              if(!gaussian_policy){
                if (std::equal(sm.a.begin(), sm.a.end(), next_action->begin()))
                  sm.pfull_data =  (1 - noise);
                else
                  sm.pfull_data = noise * 0.5f;//noise
              } else {
                sm.pfull_data = 1.f;
                for(uint i=0 ; i < nb_motors; i++)
                  sm.pfull_data *= exp(-(sm.a[i]-next_action->at(i))*(sm.a[i]-next_action->at(i))/(2.f*noise*noise));
              }
            } else 
              sm.pfull_data = 1.f;
            
            if(regularize_p0_distribution)
              sm.pfull_data /= sm.p0;
  
            delete next_action;
            
            vtraj_local.push_back(sm);
        }
      }
      
      if(!regularize_space_distribution){
        generateSubData(vtraj_final, vtraj_local, min(max_critic_batch_size, vtraj_local.size()));
        return;
      }
        
      std::vector<sample> vtraj_before_ps;
      generateSubData(vtraj_before_ps, vtraj_local, min(max_critic_batch_size, vtraj_local.size()));
      
      if(learnV){
        KDE proba_s_local;
        
        for(auto sm : vtraj_before_ps)
            proba_s_local.add_data(sm.s);
        
        proba_s.calc_bandwidth();
        
        for(auto sm : vtraj_before_ps)
            sm.pfull_data = 1 / ( 1 + proba_s_local.pdf( sm.s, proba_s.get_default_bandwidth_map(), 1.f ));
      } else {
        KDE proba_sa_local;
        
        for(auto sm : vtraj_before_ps){
            std::vector<double> sa;
            mergeSA(sa, sm.s, sm.a);
            proba_sa.add_data(sa);
            proba_sa_local.add_data(sm.s);
        }
        
        proba_sa.calc_bandwidth();
      
        for(auto sm : vtraj_before_ps){
            std::vector<double> sa;
            mergeSA(sa, sm.s, sm.a);
            sm.pfull_data = 1 / ( 1 + proba_sa_local.pdf( sa, proba_sa.get_default_bandwidth_map(), 1.f ));
        }
      }
      
      generateSubData(vtraj_final, vtraj_before_ps, min(max_critic_batch_size, vtraj_before_ps.size()));
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
        vtraj.reserve(trajectory_q_last.size() + trajectory_q.size());
        std::copy(trajectory_q.begin(), trajectory_q.end(), std::back_inserter(vtraj));
        std::copy(trajectory_q_last.begin(), trajectory_q_last.end(), std::back_inserter(vtraj));
      }
      //std::shuffle(vtraj.begin(), vtraj.end()); //useless due to rprop batch
      
      
      Critic<sample>::ParrallelTargetComputing* dq = nullptr; 
      if(!is_multi_drawn)
        dq = critic->createTargetData(vtraj, &trajectory_q_absorbing, encourage_absorbing);

      uint iteration = 0;
      auto iter = [&]() {
        if(is_multi_drawn){
          createMiniBatchV(vtraj, vtraj_precompute);
          dq = critic->createTargetData(vtraj, &trajectory_q_absorbing, encourage_absorbing);
        }
        
        if(encourage_absorbing)
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size() + trajectory_q_absorbing.size()), *dq);
        else
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), *dq);

        if(is_multi_drawn)
          critic->write_critic_file(std::to_string(episode) + "." + std::to_string(iteration));
        
        if(vnn_from_scratch)
          critic->forgotAll();
        
        if(is_multi_drawn){
          write_valuef_point_file("aaQ." + std::to_string(episode) + "." + std::to_string(iteration), dq->data);
          critic->getMLP()->learn_stoch(dq->data, 200, 0, CONVERG_PRECISION);
        }
        else
          critic->getMLP()->learn_stoch(dq->data, 10000, 0, CONVERG_PRECISION);
        
//         if(co_update){
//           update_actor();
//           vtraj_precompute.clear();
//           if(!learnV)
//             dq->update_actions();
//         }
        
        if(is_multi_drawn){
          dq->free();
          delete dq;
        }
        iteration++;
      };

      auto eval = [&]() {
        return critic->error();
      };

      bib::Converger::determinist<>(iter, eval, max_iteration, CONVERG_PRECISION, 0);

      if(!is_multi_drawn){
        dq->free();
        delete dq;
      }
      
      //       LOG_DEBUG("number of data " << trajectory.size());
    }
  }

  void save(const std::string& path) override {
    critic->getMLP()->save(path+".critic");
//     bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    critic->getMLP()->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " <<
        std::setw(8) << std::fixed << std::setprecision(5) << critic->error() << " " << noise << " " <<
        trajectory_q.size() << " " << critic->getMLP()->weightSum() << " " << trajectory_q_absorbing.size() << " "
        << trajectory_q_last.size();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << critic->error() << " " << trajectory_q.size() ;
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
  bool lecun_activation, gaussian_policy, online_actor_update,
    clear_trajectory, vnn_from_scratch, 
    importance_sampling, is_multi_drawn, encourage_absorbing, actor_update_determinist_range,
    importance_sampling_actor, learnV, regularize_space_distribution,
    sample_update, td_sample_actor_update, regularize_p0_distribution, regularize_pol_distribution;
    
  size_t max_actor_batch_size, max_critic_batch_size;
  
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
  KDE proba_sa;
  KDE proba_s_ct;
  std::set<sample> trajectory_q;
  std::set<sample> trajectory_q_last;
  std::set<sample> trajectory_a;
  std::vector<sample> trajectory_q_absorbing;

  Critic<sample>* critic;
  MLP* ann;
};

#endif

