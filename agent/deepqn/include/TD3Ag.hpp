

#ifndef TD3AG_HPP
#define TD3AG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

#include "nn/MLP.hpp"
#include "nn/DevMLP.hpp"
#include "nn/DODevMLP.hpp"
#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include "bib/OrnsteinUhlenbeckNoise.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>


#define DOUBLE_COMPARE_PRECISION 1e-9

#ifndef SAASRG_SAMPLE
#define SAASRG_SAMPLE
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
#endif

template<typename NN = MLP>
class DeepQNAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
  friend class FusionOOAg;
   
  DeepQNAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentGPUProgOptions>(_nb_motors, _nb_sensors), nb_sensors(_nb_sensors) {

  }

  virtual ~DeepQNAg() {
    delete qnn;
    delete ann;
    
    delete qnn2;
    
    delete qnn_target;
    delete ann_target;
    
    delete qnn_target2;
    
    delete ann_testing;
    
    delete hidden_unit_q;
    delete hidden_unit_a;
    
    if(oun == nullptr)
      delete oun;
    
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action = ann_testing->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning){
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached};
      insertSample(sa);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy == 1){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(gaussian_policy == 2){
        oun->step(*next_action);
      } else if(bib::Utils::rand01() < noise){ //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    ann_testing->neutral_action(sensors, next_action);
    last_action.reset(next_action);
    

    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);
    
    return *next_action;
  }

  void insertSample(const sample& sa){
    current_step_counter++;
    if(trajectory.size() >= replay_memory)
      trajectory.pop_front();
    trajectory.push_back(sa);
      
    end_episode(true);
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
    kMinibatchSize          = pt->get<uint>("agent.mini_batch_size");
    replay_memory           = pt->get<uint>("agent.replay_memory");
    tau_soft_update         = pt->get<double>("agent.tau_soft_update");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_v                 = pt->get<double>("agent.alpha_v");
    decay_v                 = pt->get<double>("agent.decay_v");
    batch_norm_critic       = pt->get<uint>("agent.batch_norm_critic");
    batch_norm_actor        = pt->get<uint>("agent.batch_norm_actor");
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    bool test_net           = pt->get<bool>("agent.test_net");
    bn_adapt                = pt->get<bool>("agent.bn_adapt");
    uint momentum           = pt->get<uint>("agent.momentum");
    //TD3 hyperparams
    policy_noise                = pt->get<double>("agent.policy_noise");
    noise_clip                = pt->get<double>("agent.noise_clip");
    policy_freq                = pt->get<int>("agent.policy_freq");
    
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

    if(gaussian_policy == 2) {
      double oun_theta = pt->get<double>("agent.noise2");
      double oun_dt = pt->get<double>("agent.noise3");
      oun = new bib::OrnsteinUhlenbeckNoise<double>(this->nb_motors, noise, oun_theta, oun_dt);
    }
    
    ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, 
                 hidden_layer_type, actor_output_layer_type, batch_norm_actor, false, momentum);
    
    if(test_net)
      ann_target = new NN(*ann, false, ::caffe::Phase::TEST);
    else
      ann_target = new NN(*ann, false);
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    ann_testing->increase_batchsize(1);
    
    qnn = new NN(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, kMinibatchSize, 
                 decay_v, hidden_layer_type, batch_norm_critic, false, momentum);
    qnn2 = new NN(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, kMinibatchSize, 
                 decay_v, hidden_layer_type, batch_norm_critic, false, momentum);
    
    if(test_net) {
      qnn_target = new NN(*qnn, false, ::caffe::Phase::TEST);
      qnn_target2 = new NN(*qnn2, false, ::caffe::Phase::TEST);
    } else {
      qnn_target = new NN(*qnn, false);
      qnn_target2 = new NN(*qnn2, false);
    }
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    if(gaussian_policy == 2)
      oun->reset();
    
    last_action = nullptr;
    last_pure_action = nullptr;
    
    current_step_counter = 0;
    
    if(std::is_same<NN, DODevMLP>::value && learning){
      DODevMLP * ann_cast = static_cast<DODevMLP *>(ann);
      bool reset_operator, changed;
      std::tie(reset_operator, changed) = ann_cast->inform(episode, last_sum_weighted_reward);
      if(reset_operator && changed){
        LOG_INFO("reset learning catched");
        trajectory.clear();
      }
      if(changed && ann_cast->ewc_enabled() && ann_cast->ewc_force_constraint()){
        static_cast<DODevMLP *>(qnn)->ewc_setup();
        //         else if(changed_vnn && !changed_ann) //impossible cause of ann structure
      }

      if(changed){
        double* weights = new double[ann->number_of_parameters(false)];
        ann->copyWeightsTo(weights, false);
        ann_testing->copyWeightsFrom(weights, false);
        delete[] weights;
      }
      //only ann and qnn shared parameters, others networks need their update
      static_cast<DODevMLP *>(qnn_target)->inform(episode, last_sum_weighted_reward);
      static_cast<DODevMLP *>(ann_target)->inform(episode, last_sum_weighted_reward);
    }
  }
  
  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
       traj[i] = from[r];
     }
  }

  void end_instance(bool learning) override {
    if(learning){
      episode++;
    }
  }
  
  void end_episode(bool learning) override {
    
    if(!learning || trajectory.size() < 250 || trajectory.size() < kMinibatchSize)
      return;

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
    
    // TD3 : learn Qpi instead of Qmu
    std::vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussianZeroMean(all_next_actions->size(), policy_noise, -noise_clip, noise_clip);
    for (int i=0;i < all_next_actions->size();i++)
        all_next_actions->at(i) = std::min(std::max(all_next_actions->at(i) + randomized_action->at(i), (double)-1.f), (double)1.f);
    delete randomized_action;
    
    //compute next q
    auto q_targets = qnn_target->computeOutVFBatch(all_next_states, *all_next_actions);
    auto q_targets2 = qnn_target2->computeOutVFBatch(all_next_states, *all_next_actions);
    delete all_next_actions;
    
    //adjust q_targets
    i=0;
    for (auto it : traj){
    if(it.goal_reached)
        q_targets->at(i) = it.r;
    else 
        q_targets->at(i) = it.r + gamma * std::min(q_targets->at(i), q_targets2->at(i));
    
    i++;
    }
    
    //Update critic
    qnn->learn_batch(all_states, all_actions, *q_targets, 1);
    qnn2->learn_batch(all_states, all_actions, *q_targets, 1);
    
    //TD3 delayed update of the actor
    if (current_step_counter % policy_freq == 0) {
        //Update actor
        qnn->ZeroGradParameters();
        ann->ZeroGradParameters();
    
        auto all_actions_outputs = ann->computeOutBatch(all_states);
        if(batch_norm_actor != 0 && bn_adapt){
            delete all_actions_outputs;
            ann_testing->increase_batchsize(kMinibatchSize);
            all_actions_outputs = ann_testing->computeOutBatch(all_states);
            ann_testing->increase_batchsize(1);
        }

        delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
        
        const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
        double* q_values_diff = q_values_blob->mutable_cpu_diff();
        i=0;
        for (auto it : traj)
            q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
        qnn->critic_backward();
        const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
        
        // Transfer input-level diffs from Critic to Actor
        const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
        actor_actions_blob->ShareDiff(*critic_action_blob);
        ann->actor_backward();
        ann->updateFisher(traj.size());
        ann->regularize();
        ann->getSolver()->ApplyUpdate();
        ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
        
        delete all_actions_outputs;
        
        // Soft update of targets networks
        qnn_target->soft_update(*qnn, tau_soft_update);
        qnn_target2->soft_update(*qnn2, tau_soft_update);
        ann_target->soft_update(*ann, tau_soft_update);
        
        double* weights = new double[ann->number_of_parameters(false)];
        ann->copyWeightsTo(weights, false);
        ann_testing->copyWeightsFrom(weights, false);
        delete[] weights;
    }
    
    delete q_targets;
    delete q_targets2;
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return qnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return nullptr;
  }

  void save(const std::string& path, bool, bool) override {
    ann->save(path+".actor");
    qnn->save(path+".critic");
    qnn2->save(path+".critic2");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    qnn->load(path+".critic");
    
    delete qnn_target;
    delete ann_target;
    
    qnn_target = new MLP(*qnn, false, ::caffe::Phase::TEST);
    ann_target = new MLP(*ann, false, ::caffe::Phase::TEST);
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
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
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
  
  uint gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint kMinibatchSize;
  uint replay_memory;
  uint batch_norm_actor, batch_norm_critic;
  uint actor_output_layer_type, hidden_layer_type;
  
  bool inverting_grad, bn_adapt;
  
  double policy_noise, noise_clip;
  int policy_freq, current_step_counter;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  
  uint episode = 0;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  
  MLP *ann, *ann_target;
  MLP *qnn, *qnn_target, *qnn2, *qnn_target2;
  MLP *ann_testing;
  
  struct algo_state {
    uint episode;
    double J;
    std::deque<std::vector<double>> best_trajectory_a;
    
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) {
      ar& BOOST_SERIALIZATION_NVP(episode);
      ar& BOOST_SERIALIZATION_NVP(J);
      ar& BOOST_SERIALIZATION_NVP(best_trajectory_a);
    }
  };
};

#endif

