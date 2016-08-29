

#ifndef DEEPQNAG_HPP
#define DEEPQNAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/filesystem.hpp>

#include "MLP.hpp"
#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>

// 
// POOL_FOR_TESTING need to be define for stochastics environements
// in order to test (learning=false) "the best" known policy
// 

#define DOUBLE_COMPARE_PRECISION 1e-9

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  double p0;
  double onpolicy_target;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(pure_a);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
    ar& BOOST_SERIALIZATION_NVP(p0);
    ar& BOOST_SERIALIZATION_NVP(onpolicy_target);
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

class AgentGPUProgOptions {
 public:
  static boost::program_options::options_description program_options() {
    boost::program_options::options_description desc("Allowed Agent options");
    desc.add_options()("load", boost::program_options::value<std::string>(), "set the agent to load");
    desc.add_options()("cpu", "use cpu [default]");
    desc.add_options()("gpu", "use gpu");
    return desc;
  }
};

class DeepQNAg : public arch::AACAgent<MLP, AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  DeepQNAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~DeepQNAg() {
    delete qnn;
    delete ann;
    
    delete qnn_target;
    delete ann_target;
    
    delete hidden_unit_q;
    delete hidden_unit_a;
    
#ifdef POOL_FOR_TESTING
    for (auto i : best_pol_population)
      delete i.ann;
#endif
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool last) override {

    vector<double>* next_action = ann->computeOut(sensors);

    if(shrink_greater_action)
      shrink_actions(next_action);
    
    if (last_action.get() != nullptr && learning){
      double p0 = 1.f;
      for(uint i=0;i < nb_motors;i++)
        p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || (count_last && last), p0, 0.};
      insertSample(sa);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = nullptr;
        if(gaussian_type == 0)
          randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        else if(gaussian_type == 1)
          randomized_action = bib::Proba<double>::multidimentionnalGaussian(*next_action, noise);
        else if(gaussian_type == 2)
          randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
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

  void insertSample(const sample& sa){
    
    if(pure_online){
      if(trajectory.size() >= replay_memory)
        trajectory.pop_front();
      trajectory.push_back(sa);
      
      end_episode();
    } else {
      last_trajectory.push_back(sa);
    }
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    kMinibatchSize          = pt->get<uint>("agent.mini_batch_size");
    pure_online             = pt->get<bool>("agent.pure_online");
    replay_memory           = pt->get<uint>("agent.replay_memory");
    inverting_grad          = pt->get<bool>("agent.inverting_grad");
    shrink_greater_action   = pt->get<bool>("agent.shrink_greater_action");
    sure_shrink 	          = pt->get<bool>("agent.sure_shrink");
    force_more_update       = pt->get<uint>("agent.force_more_update");
    tau_soft_update         = pt->get<double>("agent.tau_soft_update");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_v                 = pt->get<double>("agent.alpha_v");
    decay_v                 = pt->get<double>("agent.decay_v");
    batch_norm              = pt->get<uint>("agent.batch_norm");
    count_last              = pt->get<bool>("agent.count_last");
    gaussian_type           = pt->get<uint>("agent.gaussian_type");
    last_layer_actor        = pt->get<uint>("agent.last_layer_actor");
#ifdef POOL_FOR_TESTING
    testing_strategy        = pt->get<uint>("agent.testing_strategy");
#endif
    
#ifdef CAFFE_CPU_ONLY
    LOG_INFO("CPU mode");
    (void) command_args;
#else
    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0){
      caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
      LOG_INFO("CPU mode");
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
      caffe::Caffe::SetDevice(0);
      LOG_INFO("GPU mode");
    }
#endif    
    
    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, kMinibatchSize, decay_v, batch_norm);
    qnn_target = new MLP(*qnn, false);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, last_layer_actor, batch_norm);
    ann_target = new MLP(*ann, false);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    learning = _learning;
  }
  
  void computePTheta(vector< sample >& vtraj, double *ptheta){
    uint i=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      sample sm = *it;
      vector<double>* next_action = ann->computeOut(sm.s);
      if(shrink_greater_action)
        shrink_actions(next_action);
      
      double p0 = 1.f;
      for(uint i=0;i < nb_motors;i++)
        p0 *= exp(-(next_action->at(i)-sm.a[i])*(next_action->at(i)-sm.a[i])/(2.f*noise*noise));

      ptheta[i] = p0;
      i++;
      delete next_action;
    }
  }

  double fitfun_sum_overtraj(){
    double sum = 0;
    for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
      sample sm = *it;
      
      vector<double>* next_action = ann->computeOut(sm.s);
      if(shrink_greater_action)
        shrink_actions(next_action);
      
      sum += qnn->computeOutVF(sm.s, *next_action);
      delete next_action;
    }
    
    return sum / trajectory.size();
  }
  
  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
       traj[i] = from[r];
     }
  }
  
  void label_onpoltarget() {
    CHECK_GT(last_trajectory.size(), 0) << "Need at least one transition to label.";
    
    sample& last = last_trajectory[last_trajectory.size()-1];
    last.onpolicy_target = last.r;// Q-Val is just the final reward
    for (int i=last_trajectory.size()-2; i>=0; --i) {
      sample& t = last_trajectory[i];
      float reward = t.r;
      float target = last_trajectory[i+1].onpolicy_target;
      t.onpolicy_target = reward + gamma * target;
    }

  }
  
  void shrink_actions(vector<double>* next_action){
    for(uint i=0;i < nb_motors ; i++)
      if(next_action->at(i) > 1.f)
        next_action->at(i)=1.f;
      else if(next_action->at(i) < -1.f)
        next_action->at(i)=-1.f;
  }

#ifdef POOL_FOR_TESTING
  void start_instance(bool learning) override {
    to_be_pushed.clear();
    
    if(!learning && best_pol_population.size() > 0){
      to_be_restaured_ann = ann;
      auto it = best_pol_population.begin();
      ++it;
      if(testing_strategy <= 3)
        ann = best_pol_population.begin()->ann;
      else {
        double best_score = best_pol_population.begin()->J;
        int max_index_with_bests = -1;
        for(auto pols : best_pol_population){
          if(pols.J >= best_score)
            max_index_with_bests++;
          else
            break;
        }
        
        if(max_index_with_bests == 0)
          ann = best_pol_population.begin()->ann;
        else {
          int choosed_pol = bib::Utils::rand01() * max_index_with_bests;
          auto it = best_pol_population.begin();
          for(int i=0;i<choosed_pol;i++)
            ++it;
          ann = it->ann;
        }
      }
    } else if(learning && testing_strategy == 0){
      to_be_restaured_ann = new MLP(*ann, false);
    } else if(learning && testing_strategy == 1){
      to_be_restaured_ann = new MLP(*ann_target, false);
    }
  }
  
  void end_instance(bool learning) override {
    if(!learning && best_pol_population.size() > 0){
      //restore ann
      ann = to_be_restaured_ann;
      
      //       no cheat -> no update on testing
      //update score of played pol
//       double new_score = (best_pol_population.begin()->J * best_pol_population.begin()->played + sum_weighted_reward);
//       new_score /= ((double)best_pol_population.begin()->played + 1);
//       my_pol np = *best_pol_population.begin();
//       np.J = new_score;
//       np.played++;
//       best_pol_population.erase(best_pol_population.begin());
//       best_pol_population.insert(np);
    } else if(learning) {
      //not totaly stable because J(the policy stored here ) != sum_weighted_reward (online updates)
    
      //policies pool for testing
      if(best_pol_population.size() == 0 || best_pol_population.rbegin()->J < sum_weighted_reward){
        if(best_pol_population.size() > 200){
          //remove smallest
          auto it = best_pol_population.end();
          --it;
          delete it->ann;
          best_pol_population.erase(it);
        }
        
        if(testing_strategy <= 3 || testing_strategy == 6 ){
          MLP* pol_fitted_sample= nullptr;
          if(testing_strategy <= 1)
            pol_fitted_sample = to_be_restaured_ann;
          else if(testing_strategy == 2)
            pol_fitted_sample = new MLP(*ann, false);
          else if(testing_strategy == 3)
            pol_fitted_sample = new MLP(*ann_target, false);
          
          best_pol_population.insert({pol_fitted_sample,sum_weighted_reward, 0});
          if(testing_strategy == 6)
            best_pol_population.insert({new MLP(*ann_target, false),sum_weighted_reward, 0});
        } else {
          for(auto pol : to_be_pushed)
            best_pol_population.insert({pol,sum_weighted_reward, 0});
        }
      } else if(testing_strategy <= 1)
        delete to_be_restaured_ann;
      
    }
  }
#endif
  
  void end_episode() override {
    if(!learning || trajectory.size() < 250 || trajectory.size() < kMinibatchSize)
      return;
 
    if(!pure_online){
      while(trajectory.size() + last_trajectory.size() > replay_memory)
        trajectory.pop_front();
      
      label_onpoltarget();
      auto it = trajectory.end();
      trajectory.insert(it, last_trajectory.begin(), last_trajectory.end());
      last_trajectory.clear();

#ifdef POOL_FOR_TESTING
      dedicated_counted++;
      
      if(dedicated_counted % 20 == 0){
        if(testing_strategy == 4)
          to_be_pushed.push_back(new MLP(*ann, false));
        else if(testing_strategy == 5)
          to_be_pushed.push_back(new MLP(*ann_target, false));
      }
#endif
    }
    
    for(uint fupd=0;fupd<1+force_more_update;fupd++)
    {
    
      std::vector<sample> traj(kMinibatchSize);
      sample_transition(traj, trajectory);
      
      //compute \pi(s_{t+1})
      std::vector<double> all_next_states(traj.size() * nb_sensors);
      std::vector<double> all_states(traj.size() * nb_sensors);
      std::vector<double> all_actions(traj.size() * nb_motors);
      uint i=0;
      for (auto it : traj){
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
        std::copy(it.a.begin(), it.a.end(), all_actions.begin() + i * nb_motors);
        i++;
      }

      auto all_next_actions = ann_target->computeOutBatch(all_next_states);
      if(shrink_greater_action)
        shrink_actions(all_next_actions);
        
      //compute next q
      auto q_targets = qnn_target->computeOutVFBatch(all_next_states, *all_next_actions);
      delete all_next_actions;
      
      //adjust q_targets
      i=0;
      for (auto it : traj){
        if(it.goal_reached)
          q_targets->at(i) = it.r;
        else 
          q_targets->at(i) = it.r + gamma * q_targets->at(i);
        
        if(!pure_online)
          q_targets->at(i) = 0.5f * q_targets->at(i) + 0.5f*it.onpolicy_target;
        
        i++;
      }
      
      //Update critic
      qnn->InputDataIntoLayers(all_states.data(), all_actions.data(), q_targets->data());
      qnn->getSolver()->Step(1);
      
      //Update actor
      qnn->ZeroGradParameters();
      ann->ZeroGradParameters();
      
      auto all_actions_outputs = ann->computeOutBatch(all_states);
      if(shrink_greater_action && sure_shrink)
        shrink_actions(all_actions_outputs);

      delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
      
      const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      qnn->getNN()->BackwardFrom(qnn->GetLayerIndex(MLP::q_values_layer_name));
      const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
      
      if(inverting_grad){
        double* action_diff = critic_action_blob->mutable_cpu_diff();
        
        for (uint n = 0; n < traj.size(); ++n) {
          for (uint h = 0; h < nb_motors; ++h) {
            int offset = critic_action_blob->offset(n,0,h,0);
            double diff = action_diff[offset];
            double output = all_actions_outputs->at(offset);
            double min = -1.0; 
            double max = 1.0;
            if (diff < 0) {
              diff *= (max - output) / (max - min);
            } else if (diff > 0) {
              diff *= (output - min) / (max - min);
            }
            action_diff[offset] = diff;
          }
        }
      }
      
      // Transfer input-level diffs from Critic to Actor
      const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
      actor_actions_blob->ShareDiff(*critic_action_blob);
      ann->getNN()->BackwardFrom(ann->GetLayerIndex("action_layer"));
      ann->getSolver()->ApplyUpdate();
      ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
      
      // Soft update of targets networks
      qnn_target->soft_update(*qnn, tau_soft_update);
      ann_target->soft_update(*ann, tau_soft_update);
      
      delete q_targets;
      delete all_actions_outputs;
    }
  
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return qnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
        return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
  }

  void save(const std::string& path) override {
     ann->save(path+".actor");
     qnn->save(path+".critic");
//      bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    qnn->load(path+".critic");
    
    delete qnn_target;
    delete ann_target;
    
    qnn_target = new MLP(*qnn, false);
    ann_target = new MLP(*ann, false);
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward 
//     #ifndef NDEBUG
//     << " " << std::setw(8) << std::fixed << std::setprecision(5) << noise 
//     << " " << trajectory.size() 
//     << " " << ann->weight_l1_norm() 
//     << " " << std::fixed << std::setprecision(7) << qnn->error() 
//     << " " << qnn->weight_l1_norm()
//     #endif
    ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << trajectory.size() ;
  }
  

 private:
  uint nb_sensors;

  double noise;
  double tau_soft_update;
  double alpha_a; 
  double alpha_v;
  double decay_v;
  
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint kMinibatchSize;
  uint replay_memory;
  uint force_more_update;
  uint batch_norm;
  uint last_layer_actor;
  
  bool learning, pure_online, inverting_grad, shrink_greater_action, sure_shrink, count_last;
  uint gaussian_type;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::vector<sample> last_trajectory;

#ifdef POOL_FOR_TESTING  
  struct my_pol{
    MLP* ann;
    double J;
    uint played;
    
    bool operator< (const my_pol& b) const {
      return J > b.J;
    }
  };
  std::multiset<my_pol> best_pol_population;
  std::list<MLP*> to_be_pushed;
  MLP* to_be_restaured_ann;
  uint dedicated_counted = 0;
  uint testing_strategy;
  // 0 -> copy ann before episode
  // 1 -> copy ann_target before episode
  // 2 -> copy ann after episode
  // 3 -> copy ann_target after episode
  // 4 -> copy several ann during one episode
  // 5 -> copy several ann_target during one episode
  // 6 -> mix of 0/3
#endif  
  
  MLP* ann, *ann_target;
  MLP* qnn, *qnn_target;
};

#endif

