
#ifndef OFFVSETACFITTED_HPP
#define OFFVSETACFITTED_HPP

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


class OffVSetACFitted : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  OffVSetACFitted(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~OffVSetACFitted() {
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
    change_v_each           = pt->get<uint>("agent.change_v_each");
    strategy_u              = pt->get<uint>("agent.strategy_u");
    current_loaded_v        = pt->get<uint>("agent.index_starting_loaded_v");

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

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    //trajectory.clear();
    last_trajectory.clear();
    
    fann_reset_MSE(vnn->getNeuralNet());
    
    if(episode == 0 || episode % change_v_each == 0){
      if (! boost::filesystem::exists( "vset."+std::to_string(current_loaded_v) ) ){
        LOG_ERROR("file doesn't exists "<< "vset."+std::to_string(current_loaded_v));
        exit(1);
      }
      
      vnn->load("vset."+std::to_string(current_loaded_v));
      current_loaded_v++;
      end_episode();
    }
  }
  
  void update_actor_old_lw(){
    if (last_trajectory.size() > 0) {
      struct fann_train_data* data = fann_create_train(last_trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      std::vector<double> deltas;
      for(auto it = last_trajectory.begin(); it != last_trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
	target = sm.r;
	if (!sm.goal_reached) {
          std::vector<double> * next_action = ann->computeOut(sm.next_s);
	  double nextV = vnn->computeOutVF(sm.next_s, *next_action);
	  target += gamma * nextV;
          delete next_action;
	}
	mine = vnn->computeOutVF(sm.s, sm.a);

        if(target > mine) {
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          for (uint i = 0; i < nb_motors; i++)
            data->output[n][i] = sm.a[i];
        
          deltas.push_back(target - mine);
          n++;
        }
      }
          

      if(n > 0) {
        double* importance = new double[n];
        double norm = *std::max_element(deltas.begin(), deltas.end());
        for (uint i =0 ; i < n; i++){
            importance[i] = deltas[i] / norm;
        }    
        
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);

        ann->learn_stoch_lw(subdata, importance, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
        
        delete[] importance;
      }
      fann_destroy_train(data);
      
    } 
  }
  
  void update_actor_old_lw_all(){
    if (trajectory.size() > 0) {
      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      std::vector<double> deltas;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
	target = sm.r;
	if (!sm.goal_reached) {
          std::vector<double> * next_action = ann->computeOut(sm.next_s);
	  double nextV = vnn->computeOutVF(sm.next_s, *next_action);
	  target += gamma * nextV;
          delete next_action;
	}
	mine = vnn->computeOutVF(sm.s, sm.a);

        if(target > mine) {
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          for (uint i = 0; i < nb_motors; i++)
            data->output[n][i] = sm.a[i];
        
          deltas.push_back(target - mine);
          n++;
        }
      }
          

      if(n > 0) {
        double* importance = new double[n];
        double norm = *std::max_element(deltas.begin(), deltas.end());
        for (uint i =0 ; i < n; i++){
            importance[i] = deltas[i] / norm;
        }    
        
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);

        ann->learn_stoch_lw(subdata, importance, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
        
        delete[] importance;
      }
      fann_destroy_train(data);
      
    } 
  }

  void update_actor_old(){
     if (last_trajectory.size() > 0) {
      bool update_delta_neg;
      bool update_pure_ac;
      
      switch(strategy_u){
        case 0:
          update_pure_ac=false;
          update_delta_neg=false;
          break;
        case 1:
          update_pure_ac=true;
          update_delta_neg=false;
          break;
        case 2:
          update_pure_ac=false;
          update_delta_neg=true;
          break;
      }

      struct fann_train_data* data = fann_create_train(last_trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = last_trajectory.begin(); it != last_trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
	target = sm.r;
	if (!sm.goal_reached) {
          std::vector<double> * next_action = ann->computeOut(sm.next_s);
	  double nextV = vnn->computeOutVF(sm.next_s, *next_action);
	  target += gamma * nextV;
          delete next_action;
	}
	mine = vnn->computeOutVF(sm.s, sm.a);

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

        ann->learn_stoch(subdata, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
    }
  }
  
  void update_actor_old_all(){
     if (trajectory.size() > 0) {
      bool update_pure_ac=false;
      bool update_delta_neg=false;

      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
	target = sm.r;
	if (!sm.goal_reached) {
          std::vector<double> * next_action = ann->computeOut(sm.next_s);
	  double nextV = vnn->computeOutVF(sm.next_s, *next_action);
	  target += gamma * nextV;
          delete next_action;
	}
	mine = vnn->computeOutVF(sm.s, sm.a);

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

        ann->learn_stoch(subdata, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
    }
  }
  
   void update_actor_nfqca(){
    if(trajectory.size() > 0) {
      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);
      
      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;
        
        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = 0.f;//sm.a[i]; //don't care

        n++;
      }
      
      datann_derivative d = {vnn, (int)nb_sensors, (int)nb_motors};
    
//       auto iter = [&]() {
//         fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
//       };
// 
//       auto eval = [&]() {
//         //compute weight sum
//         return ann->weight_l1_norm();
//       };
// 
//       bib::Converger::determinist<>(iter, eval, 1, 0.0001, 25, "actor");
      fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
      
      fann_destroy_train(data);
    }
  }
  
  
  void end_episode() override {
    if(strategy_u <= 2)
      update_actor_old();
    else if(strategy_u <= 3)
      update_actor_old_lw();
    else if(strategy_u == 10)
      update_actor_nfqca();
    else if(strategy_u == 15)
      update_actor_old_lw_all();
    else if(strategy_u == 20)
      update_actor_old_all();
    else {
      LOG_ERROR("not implemented");
      exit(1); 
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
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size() ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }
  

 private:
  uint nb_sensors;
  
  uint episode = 0;
  uint current_loaded_v = 0;
  uint change_v_each;
  uint strategy_u;

  double noise;
  bool gaussian_policy, lecun_activation;
  uint hidden_unit_v;
  uint hidden_unit_a;
  
  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::set<sample> trajectory;
  std::set<sample> last_trajectory;
//     std::list<sample> trajectory;
  KDE proba_s;
  
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

