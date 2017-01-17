

#ifndef DEEPQNAG_HPP
#define DEEPQNAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

#include "nn/MLP.hpp"
#include "nn/DevMLP.hpp"
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

template<typename NN = MLP>
class DeepQNAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  DeepQNAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

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
    
    if (last_action.get() != nullptr && learning){
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || (count_last && last)};
      insertSample(sa);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
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
    if(trajectory.size() >= replay_memory)
      trajectory.pop_front();
    trajectory.push_back(sa);
      
    end_episode();
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    kMinibatchSize          = pt->get<uint>("agent.mini_batch_size");
    replay_memory           = pt->get<uint>("agent.replay_memory");
    inverting_grad          = pt->get<bool>("agent.inverting_grad");
    force_more_update       = pt->get<uint>("agent.force_more_update");
    tau_soft_update         = pt->get<double>("agent.tau_soft_update");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_v                 = pt->get<double>("agent.alpha_v");
    decay_v                 = pt->get<double>("agent.decay_v");
    batch_norm_critic       = pt->get<uint>("agent.batch_norm_critic");
    batch_norm_actor        = pt->get<uint>("agent.batch_norm_actor");
    count_last              = pt->get<bool>("agent.count_last");
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    bool test_net           = pt->get<bool>("agent.test_net");
    
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
    
    qnn = new NN(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, kMinibatchSize, decay_v, hidden_layer_type, batch_norm_critic);
    if(std::is_same<NN, DevMLP>::value)
      qnn->exploit(pt, static_cast<DeepQNAg *>(old_ag)->qnn);
    
    if(test_net)
      qnn_target = new MLP(*qnn, false);
    else
      qnn_target = new MLP(*qnn, true);

    
    ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, hidden_layer_type, actor_output_layer_type, batch_norm_actor);
    if(std::is_same<NN, DevMLP>::value)
      ann->exploit(pt, static_cast<DeepQNAg *>(old_ag)->ann);
    
    if(test_net)
      ann_target = new MLP(*ann, false);
    else
      ann_target = new MLP(*ann, true);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    learning = _learning;
  }
  
  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
       traj[i] = from[r];
     }
  }

#ifdef POOL_FOR_TESTING
  int choosenPol = 0;
  void start_instance(bool learning) override {
    
    if(!learning && best_pol_population.size() > 0){
      to_be_restaured_ann = ann;
      auto it = best_pol_population.begin();
      ++it;
      
      ann = best_pol_population.begin()->ann;
    } else if(learning){
      to_be_restaured_ann = new MLP(*ann, false);
    }
  }
  
  void end_instance(bool learning) override {
    if(!learning && best_pol_population.size() > 0){
      //restore ann
      ann = to_be_restaured_ann;
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
        
        MLP* pol_fitted_sample = to_be_restaured_ann;
        best_pol_population.insert({pol_fitted_sample,sum_weighted_reward, 0});
      } else
        delete to_be_restaured_ann;
    }
  }
#endif
  
  void end_episode() override {
    if(!learning || trajectory.size() < 250 || trajectory.size() < kMinibatchSize)
      return;
    
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
        
        i++;
      }
      
      //Update critic
      qnn->InputDataIntoLayers(all_states.data(), all_actions.data(), q_targets->data());
      qnn->getSolver()->Step(1);
      
      //Update actor
      qnn->ZeroGradParameters();
      ann->ZeroGradParameters();
      
      auto all_actions_outputs = ann->computeOutBatch(all_states);

      delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
      
      const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      qnn->critic_backward();
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
      ann->actor_backward();
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

  void save(const std::string& path, bool) override {
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
  uint batch_norm_actor, batch_norm_critic;
  uint actor_output_layer_type, hidden_layer_type;
  
  bool learning, inverting_grad, count_last;

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
  MLP* to_be_restaured_ann;
#endif  
  
  MLP* ann, *ann_target;
  MLP* qnn, *qnn_target;
};

#endif

