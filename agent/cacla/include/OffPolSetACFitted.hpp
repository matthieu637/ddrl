
#ifndef OFFPOLSETACFITTED_HPP
#define OFFPOLSETACFITTED_HPP

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
#include "kde.hpp"
#include "kdtree++/kdtree.hpp"

#define DOUBLE_COMPARE_PRECISION 1e-9

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

  //Used to store all sample into a tree, might be stochastic
  //only pure_a is negligate
  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return s[i] < b.s[i];
    }
    
    for (uint i = 0; i < a.size(); i++) {
      if(fabs(a[i] - b.a[i])>=DOUBLE_COMPARE_PRECISION)
        return a[i] < b.a[i];
    }
    
    for (uint i = 0; i < next_s.size(); i++) {
      if(fabs(next_s[i] - b.next_s[i])>=DOUBLE_COMPARE_PRECISION)
        return next_s[i] < b.next_s[i];
    }
    
    if(fabs(r - b.r)>=DOUBLE_COMPARE_PRECISION)
        return r < b.r;

    return goal_reached < b.goal_reached;
  }
  
  typedef double value_type;

  inline double operator[](size_t const N) const{
        return s[N];
  }
  
  bool same_state(const _sample& b) const{
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return false;
    }
    
    return true;
  }

} sample;


class OffPolSetACFitted : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  OffPolSetACFitted(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~OffPolSetACFitted() {
    delete kdtree_s;

    delete vnn;
    delete ann;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning){

      trajectory.insert({last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});
      last_trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});
      proba_s.add_data(last_state);
      
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached};
      
      kdtree_s->insert(sa);
      
      std::vector<double> psa;
      std::vector<double> psas;
      mergeSA(psa, last_state, *last_action);
      mergeSAS(psas, last_state, *last_action, sensors);
      
      proba_sa.add_data(psa);
      proba_sas.add_data(psas);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise){ //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
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
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    change_policy_each      = pt->get<uint>("agent.change_policy_each");
    strategy_w              = pt->get<uint>("agent.strategy_w");
    current_loaded_policy   = pt->get<uint>("agent.index_starting_loaded_policy");
    converge_precision      = pt->get<double>("agent.converge_precision");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
    
    kdtree_s = new kdtree_sample(nb_sensors);
  }

  bool learning;
  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    learning = _learning;
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    //trajectory.clear();
    last_trajectory.clear();
    
    fann_reset_MSE(vnn->getNeuralNet());
    
    if(episode == 0 || episode % change_policy_each == 0){
      if (! boost::filesystem::exists( "polset."+std::to_string(current_loaded_policy) ) ){
        LOG_ERROR("file doesn't exists "<< "polset."+std::to_string(current_loaded_policy));
        exit(1);
      }
      
      ann->load("polset."+std::to_string(current_loaded_policy));
      current_loaded_policy++;
      end_episode();
    }
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, const OffPolSetACFitted* _ptr) : vtraj(_vtraj), ptr(_ptr) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors + ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];
      }
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());
      struct fann* local_pol = fann_copy(ptr->ann->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          std::vector<double> * next_action = MLP::computeOut(local_pol, sm.next_s);
          double nextQA = MLP::computeOutVF(local_nn, sm.next_s, *next_action);
          delta += ptr->gamma * nextQA;
          delete next_action;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
      fann_destroy(local_pol);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    const OffPolSetACFitted* ptr;
  };
  
    void update_critic(){
      if (trajectory.size() > 2) {

        std::vector<sample> *vtraj; 
        double *importance_sample = nullptr;
        if(strategy_w == 0){//only on the last data
          vtraj = new std::vector<sample>(last_trajectory.size());
          std::copy(last_trajectory.begin(), last_trajectory.end(), vtraj->begin());
        } else if(strategy_w >= 1 && strategy_w <= 3){
          vtraj = new std::vector<sample>(trajectory.size());
          std::copy(trajectory.begin(), trajectory.end(), vtraj->begin());
          
          if(strategy_w > 1)//on every data
            importance_sample = new double [trajectory.size()];
          
          if(strategy_w == 2) {//on every data 1/p(s)
            uint i=0;
            for(auto it = vtraj->begin(); it != vtraj->end() ; ++it) {
              importance_sample[i] = 1.f / proba_s.pdf(it->s);
              i++;
            }
          } else if(strategy_w == 3) {//on every data 1/p(s) normalized
            uint i=0;
            double sum_ps = 0.00f;
            for(auto it = vtraj->begin(); it != vtraj->end() ; ++it) {
              sum_ps += 1.f / proba_s.pdf(it->s);
              i++;
            }
            
            i=0;
            for(auto it = vtraj->begin(); it != vtraj->end() ; ++it) {
              importance_sample[i] =  (1.f / proba_s.pdf(it->s)) / sum_ps;
              i++;
            }            
          } else if(strategy_w != 1) {
              LOG_ERROR("to be implemented");
              exit(1);
          }
          
        } else if(strategy_w != 0) {
          LOG_ERROR("to be implemented");
          exit(1);
        }
	
        ParraVtoVNext dq(*vtraj, this);

        auto iter = [&]() {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj->size()), dq);

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
          
          if(strategy_w <= 1)
            vnn->learn_stoch(dq.data, 30000, 0, converge_precision);
          else
            vnn->learn_stoch_lw(dq.data, importance_sample, 30000, 0, converge_precision);
        };

        auto eval = [&]() {
          return fann_get_MSE(vnn->getNeuralNet());
        };

        if(determinist_vnn_update)
              bib::Converger::determinist<>(iter, eval, number_fitted_iteration, converge_precision, 0);
        else {
          NN best_nn = nullptr;
          auto save_best = [&]() {
            if(best_nn != nullptr)
              fann_destroy(best_nn);
            best_nn = fann_copy(vnn->getNeuralNet());
          };

          bib::Converger::min_stochastic<>(iter, eval, save_best, 30, converge_precision, 0, 10);
          vnn->copy(best_nn);
          fann_destroy(best_nn);
        }

        dq.free(); 
        if(strategy_w != 0)
          delete[] importance_sample;
        delete vtraj;
      }
  }

  inline void mergeSA(std::vector<double>& AB, const std::vector<double>& A, const std::vector<double>& B) {
    AB.reserve( A.size() + B.size() ); // preallocate memory
    AB.insert( AB.end(), A.begin(), A.end() );
    AB.insert( AB.end(), B.begin(), B.end() );
  }
  
  inline void mergeSAS(std::vector<double>& AB, const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C) {
    AB.reserve( A.size() + B.size() + C.size() ); // preallocate memory
    AB.insert( AB.end(), A.begin(), A.end() );
    AB.insert( AB.end(), B.begin(), B.end() );
    AB.insert( AB.end(), C.begin(), C.end() );
  }

  void end_episode() override {
    if(learning)
      update_critic();
  }
  
  void learn_V(std::map<std::vector<double>, double>& bvf) override {
    if((episode+1) % change_policy_each == 0){
      trajectory.clear();
      last_trajectory.clear();
      
    
      struct fann_train_data* data = fann_create_train(bvf.size(), nb_sensors + nb_motors, 1);
      
      auto it = bvf.cbegin();
      for (uint n = 0; n < bvf.size(); n++) {
        
        for (uint i = 0; i < nb_sensors + nb_motors ; i++){
            data->input[n][i] = it->first[i];
            LOG_FILE_NNL("vset.data."+std::to_string(current_loaded_policy-1), it->first[i] << " ");
        }
        
        data->output[n][0] = it->second;
        LOG_FILE_NNL("vset.data."+std::to_string(current_loaded_policy-1), it->second << "\n");
        
        ++it;
      }
      //clear all; close all ;X=load('vset.data.904');[xx,yy] = meshgrid (linspace (-pi,pi,300));griddata(X(:,1),X(:,3),X(:,end),xx,yy);
      bib::Logger::getInstance()->closeFile("vset.data."+std::to_string(current_loaded_policy-1));
      
      fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
      vnn->learn_stoch(data, 10000, 200, 0.0000001, 200);
      
      fann_destroy_train(data);
      
      vnn->save("vset."+std::to_string(current_loaded_policy-1));
      LOG_DEBUG("vset."+std::to_string(current_loaded_policy-1) << " saved " << bvf.size() );
      bvf.clear();
    }
  }
  
  void end_instance(bool) override {
    episode++;
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
    return vnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
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
          8) << std::fixed << std::setprecision(8) << vnn->error() << " " << noise << " " << trajectory.size() ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(8) << vnn->error() << " " << trajectory.size() ;
  }
  

 private:
  uint nb_sensors;
  
  uint episode = 0;
  uint current_loaded_policy = 0;
  uint change_policy_each;
  uint strategy_w;

  double noise, converge_precision;
  bool gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update;
  uint hidden_unit_v;
  uint hidden_unit_a;
  uint number_fitted_iteration;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::set<sample> trajectory;
  std::set<sample> last_trajectory;
//     std::list<sample> trajectory;
  KDE proba_s;
  KDE proba_sa, proba_sas;
  
  struct L1_distance
  {
    typedef double distance_type;
    
    double operator() (const double& __a, const double& __b, const size_t) const
    {
      double d = fabs(__a - __b);
      return d;
    }
  };
  typedef KDTree::KDTree<sample, KDTree::_Bracket_accessor<sample>, L1_distance> kdtree_sample;
  kdtree_sample* kdtree_s;

  MLP* ann;
  MLP* vnn;
};

#endif

