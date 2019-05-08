
#ifndef FUSIONONOFFAG_HPP
#define FUSIONONOFFAG_HPP



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
//     delete from feed
//     delete ann;
    
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
//     exploration done in onpol
//     if(learning) {
//       if(gaussian_policy == 1){
//         vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
//         delete next_action;
//         next_action = randomized_action;
//       } else if(gaussian_policy == 2){
//         oun->step(*next_action);
//       } else if(bib::Utils::rand01() < noise){ //e-greedy
//         for (uint i = 0; i < next_action->size(); i++)
//           next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
//       }
//     }
//     ann_testing->neutral_action(sensors, next_action);
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
  
  void feed_ann(MLP* ann_){
      ann = ann_;
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
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

    if(gaussian_policy == 2){
      double oun_theta = pt->get<double>("agent.noise2");
      double oun_dt = pt->get<double>("agent.noise3");
      oun = new bib::OrnsteinUhlenbeckNoise<double>(this->nb_motors, noise, oun_theta, oun_dt);
    }
    
//     initialized from feed
//     ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, 
//                  hidden_layer_type, actor_output_layer_type, batch_norm_actor, false, momentum);
    
    if(test_net)
      ann_target = new NN(*ann, false, ::caffe::Phase::TEST);
    else
      ann_target = new NN(*ann, false);
    ann_target->increase_batchsize(kMinibatchSize);
    
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
  }
  
  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
       traj[i] = from[r];
     }
  }

  void end_instance(bool learning) override {
    if(learning)
      episode++;
  }
  
  void end_episode(bool learning) override {
    
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
      
      //Update actor
      qnn->ZeroGradParameters();
//       ann->ZeroGradParameters();
      
      //TD3 delayed update of the actor
      if (current_step_counter % policy_freq == 0) {
//         auto all_actions_outputs = ann->computeOutBatch(all_states);
//         if(batch_norm_actor != 0 && bn_adapt){
//             delete all_actions_outputs;
//             ann_testing->increase_batchsize(kMinibatchSize);
//             all_actions_outputs = ann_testing->computeOutBatch(all_states);
//             ann_testing->increase_batchsize(1);
//         }
// 
//         delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
//         
//         const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
//         double* q_values_diff = q_values_blob->mutable_cpu_diff();
//         i=0;
//         for (auto it : traj)
//             q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
//         qnn->critic_backward();
//         const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
//         
//         // Transfer input-level diffs from Critic to Actor
//         const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
//         actor_actions_blob->ShareDiff(*critic_action_blob);
//         ann->actor_backward();
//         ann->updateFisher(traj.size());
//         ann->regularize();
//         ann->getSolver()->ApplyUpdate();
//         ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
//         
//         delete all_actions_outputs;
        
        // Soft update of targets networks
        qnn_target->soft_update(*qnn, tau_soft_update);
        qnn_target2->soft_update(*qnn2, tau_soft_update);
        ann_target->soft_update(*ann, tau_soft_update);
      }
      
      delete q_targets;
      delete q_targets2;
    }
    
    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return qnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return nullptr;
  }

  void save(const std::string& path, bool save_best, bool) override {
    ann->save(path+".actor");
    qnn->save(path+".critic");
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
  uint force_more_update;
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
};


template<typename NN = MLP>
class OfflineCaclaAg : public arch::AACAgent<NN, arch::AgentProgOptions> {
 public:
  typedef NN PolicyImpl;
  friend class FusionOOAg;

  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<NN, arch::AgentProgOptions>(_nb_motors, _nb_sensors), nb_sensors(_nb_sensors), empty_action(0) {

  }

  virtual ~OfflineCaclaAg() {
    delete vnn;
    delete ann;
    
    delete ann_testing;
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
    
    if(oun == nullptr)
      delete oun;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action = ann_testing->computeOut(sensors);
    if (last_action.get() != nullptr && learning)
      trajectory.push_back( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy == 1) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(gaussian_policy == 2) {
        oun->step(*next_action);
      } else if(gaussian_policy == 3 && bib::Utils::rand01() < noise2) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(gaussian_policy == 4) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise * pow(noise2, noise3 - ((double) step)));
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise) { //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);

    ann_testing->neutral_action(sensors, next_action);

    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    step++;
    
    return *next_action;
  }

  void feed_qnn(MLP *qnn_, MLP *qnn_target_, MLP *qnn2_, MLP *qnn_target2_) {
      qnn = qnn_;
      qnn2 = qnn2_;
      qnn_target = qnn_target_ ;
      qnn_target2 = qnn_target2_;
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    update_critic_first     = pt->get<bool>("agent.update_critic_first");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");
    stoch_iter_actor        = pt->get<uint>("agent.stoch_iter_actor");
    stoch_iter_critic       = pt->get<uint>("agent.stoch_iter_critic");
    batch_norm_actor        = pt->get<uint>("agent.batch_norm_actor");
    batch_norm_critic       = pt->get<uint>("agent.batch_norm_critic");
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_v                 = pt->get<double>("agent.alpha_v");
    lambda                  = pt->get<double>("agent.lambda");
    momentum                = pt->get<uint>("agent.momentum");
    beta_target             = pt->get<double>("agent.beta_target");
    ignore_poss_ac          = pt->get<bool>("agent.ignore_poss_ac");
    conserve_beta           = pt->get<bool>("agent.conserve_beta");
    idea_min_delta          = pt->get<bool>("agent.idea_min_delta");
    idea_target_qnn        = pt->get<bool>("agent.idea_target_qnn");
    idea_min_qnn            = pt->get<bool>("agent.idea_min_qnn");
    control_valid_fusion = pt->get<bool>("agent.control_valid_fusion");
    gae                     = false;
    update_each_episode = 1;
    
    if(gaussian_policy == 2){
      double oun_theta = pt->get<double>("agent.noise2");
      double oun_dt = pt->get<double>("agent.noise3");
      oun = new bib::OrnsteinUhlenbeckNoise<double>(this->nb_motors, noise, oun_theta, oun_dt);
    } else if (gaussian_policy == 3){
      noise2 = pt->get<double>("agent.noise2");
    } else if (gaussian_policy == 4){
      noise2 = pt->get<double>("agent.noise2");
      noise3 = pt->get<double>("agent.noise3");
    }
    
    try {
      update_each_episode     = pt->get<uint>("agent.update_each_episode");
    } catch(boost::exception const& ) {
    }
    
    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");
    
    if(lambda >=0. && batch_norm_critic != 0){
      LOG_DEBUG("to be done!");
      exit(1);
    }
    
    ann = new NN(nb_sensors, *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type, batch_norm_actor, true, momentum);
    
    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, false, momentum);
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    if(batch_norm_critic != 0)
      vnn_testing = new NN(*vnn, false, ::caffe::Phase::TEST);
    
    if(std::is_same<NN, DODevMLP>::value){
      try {
        if(pt->get<bool>("devnn.reset_learning_algo")){
          LOG_ERROR("NFAC cannot reset anything with DODevMLP");
          exit(1);
        }
      } catch(boost::exception const& ) {
      }
    }
    
    bestever_score = std::numeric_limits<double>::lowest();
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    step = 0;
    if(gaussian_policy == 2)
      oun->reset();
    
    if(std::is_same<NN, DODevMLP>::value && learning){
      DODevMLP * ann_cast = static_cast<DODevMLP *>(ann);
      bool changed_ann = std::get<1>(ann_cast->inform(episode, this->last_sum_weighted_reward));
//       don't need to inform vnn because they share parameters
//       static_cast<DODevMLP *>(vnn)->inform(episode, this->last_sum_weighted_reward);
      if(changed_ann && ann_cast->ewc_enabled() && ann_cast->ewc_force_constraint()){
        static_cast<DODevMLP *>(vnn)->ewc_setup();
//         else if(changed_vnn && !changed_ann) //impossible cause of ann structure
      }
    }
    
    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
  }
  
  void critic_qnn(std::vector<double>& deltas_off, const std::vector<double>& sensors, const std::vector<double>& actions, const std::vector<double>& pure_actions){      
    auto qnn1_ = qnn;
    auto qnn2_ = qnn2;
    if (idea_target_qnn){
            qnn1_ = qnn_target;
            qnn2_ = qnn_target2;
    }
    
    int kmbs = qnn1_->get_batchsize();
    qnn1_->increase_batchsize(trajectory.size());
    auto qsa = qnn1_->computeOutVFBatch(sensors, actions);
    auto vsa = qnn1_->computeOutVFBatch(sensors, pure_actions);
    if (idea_min_qnn) {
        qnn2_->increase_batchsize(trajectory.size());
        auto qsa2 = qnn2_->computeOutVFBatch(sensors, actions);
        auto vsa2 = qnn2_->computeOutVFBatch(sensors, pure_actions);
        
        for(int i=0;i<trajectory.size();i++) {
            deltas_off[i] = std::min(qsa->at(i), qsa2->at(i)) - std::max(vsa->at(i), vsa2->at(i));
        }
    } else {
        for(int i=0;i<trajectory.size();i++){
            deltas_off[i] = qsa->at(i) - vsa->at(i);
        }
    }
    qnn1_->increase_batchsize(kmbs);
    if (idea_min_qnn)
        qnn2_->increase_batchsize(kmbs);
  }

  double update_critic() {
    double V_pi_s0 = 0.f;
    if (trajectory.size() > 0) {
      //remove trace of old policy
      auto iter = [&]() {
        std::vector<double> all_states(trajectory.size() * nb_sensors);
        std::vector<double> all_next_states(trajectory.size() * nb_sensors);
        std::vector<double> v_target(trajectory.size());
        int li=0;
        for (auto it : trajectory) {
          std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
          std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
          li++;
        }

        decltype(vnn_testing->computeOutVFBatch(all_next_states, empty_action)) all_nextV;
        if(batch_norm_critic != 0)
        {
          double* weights = new double[vnn->number_of_parameters(false)];
          vnn->copyWeightsTo(weights, false);
          vnn_testing->copyWeightsFrom(weights, false);
          delete[] weights;
          all_nextV = vnn_testing->computeOutVFBatch(all_next_states, empty_action);
        } else 
          all_nextV = vnn->computeOutVFBatch(all_next_states, empty_action);

        li=0;
        for (auto it : trajectory) {
          double target = it.r;
          if (!it.goal_reached) {
            double nextV = all_nextV->at(li);
            target += this->gamma * nextV;
          }

          v_target[li] = target;
          li++;
        }

        ASSERT((uint)li == trajectory.size(), "");
        if(vnn_from_scratch){
          delete vnn;
          vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, trajectory.size(), -1, hidden_layer_type, batch_norm_critic, false, momentum);
        }
        if(lambda < 0.f && batch_norm_critic == 0)
          vnn->learn_batch(all_states, empty_action, v_target, stoch_iter_critic);
        else if(lambda < 0.f){
          for(uint sia = 0; sia < stoch_iter_critic; sia++){
            delete vnn->computeOutVFBatch(all_states, empty_action);
            {
              double* weights = new double[vnn->number_of_parameters(false)];
              vnn->copyWeightsTo(weights, false);
              vnn_testing->copyWeightsFrom(weights, false);
              delete[] weights;
            }
            auto all_V = vnn_testing->computeOutVFBatch(all_states, empty_action);
            
            const auto q_values_blob = vnn->getNN()->blob_by_name(MLP::q_values_blob_name);
            double* q_values_diff = q_values_blob->mutable_cpu_diff();
            uint i=0;
            double s = trajectory.size();
            for (auto it : trajectory){
              q_values_diff[i] = (all_V->at(i)-v_target[i])/s;
              i++;
            }
            vnn->critic_backward();
            vnn->updateFisher(trajectory.size());
            vnn->regularize();
            vnn->getSolver()->ApplyUpdate();
            vnn->getSolver()->set_iter(vnn->getSolver()->iter() + 1);
            delete all_V;
          }
        }
        else {
          auto all_V = vnn->computeOutVFBatch(all_states, empty_action);
          std::vector<double> deltas(trajectory.size());
//           
//        Simple computation for lambda return
//           
//          bib::Logger::PRINT_ELEMENTS(*all_V, "V0 ");
          li=0;
          for (auto it : trajectory){
            deltas[li] = v_target[li] - all_V->at(li);
            ++li;
          }
//          bib::Logger::PRINT_ELEMENTS(deltas, "deltas ");
          
          std::vector<double> diff(trajectory.size());
          li=trajectory.size() - 1;
          double prev_delta = 0.;
          int index_ep = trajectory_end_points.size() - 1;
          for (auto it : trajectory) {
            if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
                prev_delta = 0.;
                index_ep--;
            }
            diff[li] = deltas[li] + prev_delta;
            prev_delta = this->gamma * lambda * diff[li];
            --li;
          }
          ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");
          
          li=0;
          for (auto it : trajectory){
            diff[li] = diff[li] + all_V->at(li);
            ++li;
          }
          
          V_pi_s0 = diff[0];
          vnn->learn_batch(all_states, empty_action, diff, stoch_iter_critic);
          delete all_V;
        }
        
        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();
    }
    
    return V_pi_s0;
  }

  void end_episode(bool learning) override {
//     LOG_FILE("policy_exploration", ann->hash());
    if(!learning){
      return;
    }
    
    //learning phase
    trajectory_end_points.push_back(trajectory.size());
    if (episode % update_each_episode != 0)
      return;

    if(trajectory.size() > 0){
      vnn->increase_batchsize(trajectory.size());
      if(batch_norm_critic != 0)
        vnn_testing->increase_batchsize(trajectory.size());
    }
    
    double V_pi_s0 = 0;
    if(update_critic_first)
      V_pi_s0 = update_critic();

    if (trajectory.size() > 0) {
      const std::vector<bool> disable_back_ac(this->nb_motors, true);
      std::vector<double> deltas(trajectory.size());

      std::vector<double> all_states(trajectory.size() * nb_sensors);
      std::vector<double> all_next_states(trajectory.size() * nb_sensors);
      int li=0;
      for (auto it : trajectory) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
        li++;
      }

      decltype(vnn->computeOutVFBatch(all_next_states, empty_action)) all_nextV, all_mine;
      if(batch_norm_critic != 0)
      {
        double* weights = new double[vnn->number_of_parameters(false)];
        vnn->copyWeightsTo(weights, false);
        vnn_testing->copyWeightsFrom(weights, false);
        delete[] weights;
        all_nextV = vnn_testing->computeOutVFBatch(all_next_states, empty_action);
        all_mine = vnn_testing->computeOutVFBatch(all_states, empty_action);
      } else {
        all_nextV = vnn->computeOutVFBatch(all_next_states, empty_action);
        all_mine = vnn->computeOutVFBatch(all_states, empty_action);
      }
      
      li=0;
      for (auto it : trajectory){
        sample sm = it;
        double v_target = sm.r;
        if (!sm.goal_reached) {
          double nextV = all_nextV->at(li);
          v_target += this->gamma * nextV;
        }
        
        deltas[li] = v_target - all_mine->at(li);
        ++li;
      }
      
      if(gae){
        //           
        //        Simple computation for lambda return
        //           
        std::vector<double> diff(trajectory.size());
        li=trajectory.size() - 1;
        double prev_delta = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it : trajectory) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              index_ep--;
          }
          diff[li] = deltas[li] + prev_delta;
          prev_delta = this->gamma * lambda * diff[li];
          --li;
        }
        ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");

        li=0;
        for (auto it : trajectory){
//           diff[li] = diff[li] + all_V->at(li);
          deltas[li] = diff[li];
          ++li;
        }
      }
      
      uint n=0, n2 = 0, n3=0;
      std::vector<double> sensors(2*trajectory.size() * nb_sensors);
      std::vector<double> actions(2*trajectory.size() * this->nb_motors);
      std::vector<double> pure_actions(trajectory.size() * this->nb_motors);
      std::vector<bool> disable_back(2*trajectory.size() * this->nb_motors, false);
      std::vector<double> deltas_blob(trajectory.size() * this->nb_motors);
      
      std::vector<double> deltas_off(trajectory.size());
      li=0;
      //cacla cost
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        std::copy(it->a.begin(), it->a.end(), actions.begin() + li * this->nb_motors);
        std::copy(it->pure_a.begin(), it->pure_a.end(), pure_actions.begin() + li * this->nb_motors);
        std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
        li++;
      }
      if (!control_valid_fusion) {
        critic_qnn(deltas_off, sensors, actions, pure_actions);
      } else {
        for(int i=0;i<deltas_off.size();i++)
            deltas_off[i] = 1.f;
        
        //keep only 25% best actions
        int good_actions=0;
        for(int i=0;i<deltas.size();i++)
            if(deltas[i] > 0.)
                good_actions++;
        
        double ratio_of_good_actions=((double) good_actions)/((double)deltas.size());
        if (ratio_of_good_actions > 0.25f) {
            int number_good_action_to_remove = (ratio_of_good_actions - 0.25f)*deltas.size();
            
            std::vector<double> sorted_deltas(deltas);
            std::sort(sorted_deltas.begin(), sorted_deltas.end());
            for(int i=0;i<deltas.size();i++)
                if(deltas[i] <= sorted_deltas[number_good_action_to_remove])
                    deltas[i] = -1.f;
        }
      }
      //cacla cost
      li=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        if(deltas[li] > 0.)
            n2++;
        if(deltas_off[li] > 0.)
            n3++;
        
        if(deltas[li] > 0. && deltas_off[li] > 0.) {
//        if(deltas[li] > 0.) {
          n++;
        } else {
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        li++;
      }
      
      if(idea_min_delta){
          for(int i=0;i<deltas.size();i++)
              deltas[i]=std::min(deltas[i], deltas_off[i]);
      }
      
      //penalty cost
      int li2=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        std::copy(it->pure_a.begin(), it->pure_a.end(), actions.begin() + li * this->nb_motors);
        if(ignore_poss_ac && deltas[li2] > 0. && deltas_off[li2] > 0.) {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        li++;
        li2++;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      ratio_valid_advantage2 = ((float)n2) / ((float) trajectory.size());
      ratio_valid_advantage3 = ((float)n3) / ((float) trajectory.size());
      
      double beta=1.f;
      if(conserve_beta)
        beta=conserved_beta;

      if(n > 0) {
        for(uint sia = 0; sia < stoch_iter_actor; sia++){
          ann->increase_batchsize(2*trajectory.size());
          //learn BN
          auto ac_out = ann->computeOutBatch(sensors);
          if(batch_norm_actor != 0) {
            //re-compute ac_out with BN as testing
            double* weights = new double[ann->number_of_parameters(false)];
            ann->copyWeightsTo(weights, false);
            ann_testing->copyWeightsFrom(weights, false);
            delete[] weights;
            delete ac_out;
            ann_testing->increase_batchsize(2*trajectory.size());
            ac_out = ann_testing->computeOutBatch(sensors);
          }
          ann->ZeroGradParameters();
          
          //compute deter distance(pi, pi_old)
          double l2distance = 0.;
          int size_cost_cacla=trajectory.size()*this->nb_motors;
          for(int i=size_cost_cacla;i<actions.size();i++) {
              double x = actions[i] - ac_out->at(i);
              l2distance += x*x;
          }
          l2distance = std::sqrt(l2distance/((double) trajectory.size()*this->nb_motors));
          if (l2distance < beta_target/1.5)
              beta = beta/2.;
          else if (l2distance > beta_target*1.5)
              beta = beta*2.;
          else if (sia > 0)
              break;
          
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          auto ac_diff = actor_actions_blob->mutable_cpu_diff();
          
          for(int i=0; i<actor_actions_blob->count(); i++) {
            if(disable_back[i]) {
            ac_diff[i] = 0.00000000f;
            } else {
                double x = actions[i] - ac_out->at(i);
                if(i < size_cost_cacla)
                    ac_diff[i] = -x * deltas_blob[i];
                else
                    ac_diff[i] = -x * beta;
            }
          }
          ann->actor_backward();
          ann->updateFisher(n);
          ann->regularize();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
          delete ac_out;
        }
      } else if(batch_norm_actor != 0){
        ann->increase_batchsize(trajectory.size());
        delete ann->computeOutBatch(sensors);
      }
      
      conserved_beta = beta;

      delete all_nextV;
      delete all_mine;
      
      if(batch_norm_actor != 0)
        ann_testing->increase_batchsize(1);
    }

    if(!update_critic_first){
      update_critic();
    }
    
    nb_sample_update= trajectory.size();
    trajectory.clear();
    trajectory_end_points.clear();
    
    ann->ewc_decay_update();
    vnn->ewc_decay_update();
  }

  void end_instance(bool learning) override {
    if(learning)
      episode++;
  }

  void save(const std::string& path, bool savebest, bool learning) override {
    if(savebest) {
      if(!learning && this->sum_weighted_reward >= bestever_score) {
        bestever_score = this->sum_weighted_reward;
        ann->save(path+".actor");
      }
    } else {
      ann->save(path+".actor");
      vnn->save(path+".critic");
    } 
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

  double criticEval(const std::vector<double>&, const std::vector<double>&) override {
    LOG_INFO("not implemented");
    return 0;
  }

  arch::Policy<NN>* getCopyCurrentPolicy() override {
//         return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
    return nullptr;
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward/this->gamma << " " << this->sum_reward << 
        " " << std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << nb_sample_update <<
          " " << std::setprecision(3) << ratio_valid_advantage << " " << vnn->weight_l1_norm() << " " << ann->weight_l1_norm();
  }

  void _dump(std::ostream& out) const override {
    out << std::setprecision(5) << vnn->error() << " " << nb_sample_update << " " << std::setprecision(3) << ratio_valid_advantage 
        << " " << ratio_valid_advantage2 << " "  << ratio_valid_advantage3  << " " << vnn->weight_l1_norm() << " " << ann->weight_l1_norm();
  }
  
 private:
  uint nb_sensors;
  uint episode = 1;
  uint step = 0;

  double noise, noise2, noise3;
  uint gaussian_policy;
  bool vnn_from_scratch, update_critic_first, gae, ignore_poss_ac, conserve_beta;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, beta_target;
  double conserved_beta= 1.f;
  bool idea_min_delta, idea_target_qnn, idea_min_qnn, control_valid_fusion;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<sample> trajectory;
  std::deque<int> trajectory_end_points;

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* vnn_testing;
  
  MLP *qnn, *qnn_target, *qnn2, *qnn_target2;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  float ratio_valid_advantage=0, ratio_valid_advantage2 = 0, ratio_valid_advantage3 = 0;
  int nb_sample_update = 0;
  
};


class FusionOOAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  FusionOOAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentGPUProgOptions>(_nb_motors, _nb_sensors), 
        offpolicy_ag(_nb_motors, _nb_sensors), onpolicy_ag(_nb_motors, _nb_sensors) {

  }

  virtual ~FusionOOAg() {
    
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) override {
        
        if(! learning)
            return onpolicy_ag.runf(reward, sensors, learning, goal_reached, false);
        
        const std::vector<double>& onpol_ac = onpolicy_ag.runf(reward, sensors, learning, goal_reached, false);
        offpolicy_ag.runf(reward, sensors, learning, goal_reached, false);
        
        //copy PeNFAC action to TD3 explo
        std::copy(onpol_ac.begin(), onpol_ac.end(), offpolicy_ag.last_action->begin());
        return onpol_ac;
  }
  
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    //hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    onpolicy_ag.unique_invoke(pt, command_args, false);
    offpolicy_ag.feed_ann(onpolicy_ag.ann);
    
    boost::property_tree::ptree* properties = new boost::property_tree::ptree;
        boost::property_tree::ini_parser::read_ini("config.off.ini", *properties);    
    offpolicy_ag.unique_invoke(properties, command_args, false);
    delete properties;
    
    onpolicy_ag.feed_qnn(offpolicy_ag.qnn, offpolicy_ag.qnn2, offpolicy_ag.qnn_target, offpolicy_ag.qnn_target2);
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    onpolicy_ag.start_episode(sensors, learning);
    offpolicy_ag.start_episode(sensors, learning);
  }

  void end_instance(bool learning) override {
    onpolicy_ag.end_instance(learning);
    offpolicy_ag.end_instance(learning);
  }
  
  void end_episode(bool learning) override {
    onpolicy_ag.end_episode(learning);
    offpolicy_ag.end_episode(learning);
  }

  void save(const std::string& path, bool save_best, bool) override {

  }

  void load(const std::string& path) override {

  }
  
  void save_run() override {
 
  }
  
  void load_previous_run() override {

  }
  
    double criticEval(const std::vector<double>&, const std::vector<double>&) override {
    LOG_INFO("not implemented");
    return 0;
  }
  
    arch::Policy<MLP>* getCopyCurrentPolicy() override {
//         return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
    return nullptr;
  }

 protected:
  void _display(std::ostream& out) const override {
     onpolicy_ag._display(out);
     out << " ";
     offpolicy_ag._display(out);
  }

//clear all; close all; wndw = 10; X=load('0.learning.data'); X=filter(ones(wndw,1)/wndw, 1, X); startx=0; starty=800; width=400; height=350; figure('position',[startx,starty,width,height]); plot(X(:,3), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('sum rewards', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,2), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('steps', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,6), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('valid adv both', "fontsize", 16); ylim([0, 1]);  startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,7), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('valid adv on', "fontsize", 16); ylim([0, 1]); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,8), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('valid adv off', "fontsize", 16);ylim([0, 1]); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,9), "linewidth", 2); hold on; plot(X(:,10), "linewidth", 2, "color", "red"); legend("critic", "actor"); xlabel('learning episode', "fontsize", 16); ylabel('||\theta||_1', "fontsize", 16);
  void _dump(std::ostream& out) const override {
     onpolicy_ag._dump(out);
     out << " ";
     offpolicy_ag._dump(out);
  }

 private:
  DeepQNAg<MLP> offpolicy_ag;
  OfflineCaclaAg<MLP> onpolicy_ag;
};

#endif

