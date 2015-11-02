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
#define CONVERG_PRECISION 0.0001

#ifndef NDEBUG
// #define DEBUG_FILE
#endif

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
              p0 = noise * 0.5f;
        } else {
            p0 = 1.f;
            for(uint i=0;i < nb_motors;i++)
              p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
        }
      }
      
      sample sm = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0};
      
      double exploration = critic->evaluateExploration(sm);
      double exploitation = critic->evaluateExploitation(sm);
      
      if(exploration > exploitation){
          ann->learn(last_state, *last_action);
      }

      trajectory_q_last.insert(sm);
      proba_s.add_data(last_state);
      
      std::vector<double> sa;
      mergeSA(sa, last_state, *last_action);
      proba_sa.add_data(sa);
      
      if(goal_reached)
        if(encourage_absorbing)
          trajectory_q_absorbing.push_back( sm);
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
    alpha_a                             = pt->get<double>("agent.alpha_a");
    
    vnn_from_scratch                    = pt->get<bool>("agent.vnn_from_scratch");
    
    lecun_activation                    = pt->get<bool>("agent.lecun_activation");
    gaussian_policy                     = pt->get<bool>("agent.gaussian_policy");
    clear_trajectory                    = pt->get<bool>("agent.clear_trajectory");
    importance_sampling                 = pt->get<bool>("agent.importance_sampling");
    is_multi_drawn                      = pt->get<bool>("agent.is_multi_drawn");
    encourage_absorbing                 = pt->get<bool>("agent.encourage_absorbing");
    learnV                              = pt->get<bool>("agent.learnV");
    regularize_space_distribution       = pt->get<bool>("agent.regularize_space_distribution");
    regularize_p0_distribution          = pt->get<bool>("agent.regularize_p0_distribution");
    regularize_pol_distribution         = pt->get<bool>("agent.regularize_pol_distribution");
    
    
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
    
    episode=-1;
    
    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, alpha_a, lecun_activation);
    else {
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
      fann_set_learning_rate(ann->getNeuralNet(), alpha_a);
      fann_set_training_algorithm(ann->getNeuralNet(), FANN_TRAIN_INCREMENTAL);
    }
    
    critic = new Critic<sample>(nb_sensors, nb_motors, learnV, hidden_unit_v, lecun_activation, ann, gamma);
    
    fann_set_learning_rate(ann->getNeuralNet(), alpha_a / fann_get_total_connections(ann->getNeuralNet()));
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
    
    episode ++;
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
      
      update_critic();
           
      for (auto sm : trajectory_q_last)
        trajectory_q.insert(sm);
      
//       LOG_DEBUG(ann->weightSum());
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
  void write_policy_addeddata(const std::string& , vector< sample > , const MLP& ){
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
  void write_valuef_point_file(const std::string& , fann_train_data* ){
#endif
  }
  
#ifdef DEBUG_FILE
  void write_policy_data(const std::string& file, const std::vector<double>& s, 
                         const std::vector<double>& a, double explorMexploi){
      LOG_FILE(file, s[0] << " " << a[0] << " " << explorMexploi);
#else
  void write_policy_data(const std::string& , const std::vector<double>& , 
                         const std::vector<double>& , double){
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
                  sm.pfull_data =  (1.f - noise);
                else
                  sm.pfull_data = noise * 0.5f;
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
        vtraj.reserve(trajectory_q_last.size());
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
  bool lecun_activation, gaussian_policy,
    clear_trajectory, vnn_from_scratch, 
    importance_sampling, is_multi_drawn, encourage_absorbing,
    learnV, regularize_space_distribution, 
    regularize_p0_distribution, regularize_pol_distribution;
    
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
  std::vector<sample> trajectory_q_absorbing;

  Critic<sample>* critic;
  MLP* ann;
};

#endif

