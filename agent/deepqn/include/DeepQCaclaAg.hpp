
#ifndef DEEPQCACLA_HPP
#define DEEPQCACLA_HPP

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
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>


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
class DeepQCaclaAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  DeepQCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~DeepQCaclaAg() {
    delete qnn;
    delete ann;
    
    delete qnn_target;
    delete ann_target;
    
    delete ann_testing;
    
    delete hidden_unit_q;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
    vector<double>* next_action = ann_testing->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning){
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached};
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

    last_trajectory_a.push_back(*next_action);
    
    return *next_action;
  }

  void insertSample(const sample& sa){
    if(trajectory.size() >= replay_memory)
      trajectory.pop_front();
    trajectory.push_back(sa);
      
    end_episode(true);
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
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    bool test_net           = pt->get<bool>("agent.test_net");
    bn_adapt                = pt->get<bool>("agent.bn_adapt");
    uint momentum           = pt->get<uint>("agent.momentum");
    qac_target              = pt->get<bool>("agent.qac_target");
    aac_target              = pt->get<bool>("agent.aac_target");
    qac_sample              = pt->get<uint>("agent.qac_sample");
    qnextac_sample          = pt->get<uint>("agent.qnextac_sample");
    recompute_next_ac       = pt->get<bool>("agent.recompute_next_ac");
    onpolac                 = pt->get<bool>("agent.onpolac");
    qmu                     = pt->get<bool>("agent.qmu");
    
    if(recompute_next_ac && !aac_target) {
      LOG_DEBUG("recompute_next_ac useless");
      exit(1);
    }
    
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
    
    ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, 
                 hidden_layer_type, actor_output_layer_type, batch_norm_actor, false, momentum);
    if(std::is_same<NN, DevMLP>::value)
      ann->exploit(pt, static_cast<DeepQCaclaAg *>(old_ag)->ann);
    else if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);
    
    if(test_net)
      ann_target = new NN(*ann, false, ::caffe::Phase::TEST);
    else
      ann_target = new NN(*ann, false);
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    ann_testing->increase_batchsize(1);
    
    qnn = new NN(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, kMinibatchSize, 
                 decay_v, hidden_layer_type, batch_norm_critic, false, momentum);
    if(std::is_same<NN, DevMLP>::value)
      qnn->exploit(pt, static_cast<DeepQCaclaAg *>(old_ag)->qnn);
    else if(std::is_same<NN, DODevMLP>::value)
      qnn->exploit(pt, ann);
    
    if(test_net)
      qnn_target = new NN(*qnn, false, ::caffe::Phase::TEST);
    else
      qnn_target = new NN(*qnn, false);
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    last_trajectory_a.clear();
    
    if(std::is_same<NN, DODevMLP>::value){
      if(static_cast<DODevMLP *>(qnn)->inform(episode, last_sum_weighted_reward)){
        LOG_INFO("reset learning catched");
        trajectory.clear();
      }
      if(static_cast<DODevMLP *>(ann)->inform(episode, last_sum_weighted_reward)){
        LOG_INFO("reset learning catched");
        trajectory.clear();
      }
      static_cast<DODevMLP *>(qnn_target)->inform(episode, last_sum_weighted_reward);
      static_cast<DODevMLP *>(ann_target)->inform(episode, last_sum_weighted_reward);
      static_cast<DODevMLP *>(ann_testing)->inform(episode, last_sum_weighted_reward);
    }
  }
  
  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
       traj[i] = from[r];
     }
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
      std::vector<bool> disable_back(traj.size() * this->nb_motors, false);
      const std::vector<bool> disable_back_ac(this->nb_motors, true);
      std::vector<double> deltas(traj.size());
      uint i=0;
      for (auto it : traj){
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
        std::copy(it.a.begin(), it.a.end(), all_actions.begin() + i * nb_motors);
        i++;
      }

      auto all_next_actions = ann_target->computeOutBatch(all_next_states);
        
      //compute next q
      std::vector<double>* q_targets;
      if(!qmu)
        q_targets = qnn_target->computeOutVFBatch(all_next_states, *all_next_actions);
      else {
        q_targets = new std::vector<double>(traj.size(), 0.f);
        for(uint j=0;j<7;j++){
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*all_next_actions, noise);
          auto local_all_nextV = qnn_target->computeOutVFBatch(all_next_states, *randomized_action);
          std::transform(q_targets->begin(), q_targets->end(), local_all_nextV->begin(), q_targets->begin(), std::plus<double>());
          delete local_all_nextV;
          delete randomized_action;
        }
        double factor_ = 1.f/((double)7.);
        std::transform(q_targets->begin(), q_targets->end(), q_targets->begin(), std::bind1st(std::multiplies<double>(), factor_));
      }
      
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
      qnn->stepCritic(all_states, all_actions, *q_targets);
      
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
      
      MLP* qnn_worker = qnn;
      if(qac_target)
        qnn_worker = qnn_target;
      
      MLP* ann_worker = ann;
      if(aac_target)
        ann_worker = ann_target;
      
      // pre-compute delta
      std::vector<double>* all_mine, *all_nextV;
      
      std::vector<double>* current_all_actions = all_actions_outputs;
      if(onpolac)
        current_all_actions = &all_actions;
      
      if(qac_sample <= 1)
        all_mine = qnn_worker->computeOutVFBatch(all_states, *current_all_actions);
      else {
        all_mine = new std::vector<double>(traj.size(), 0.f);
        for(uint j=0;j<qac_sample;j++){
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*current_all_actions, noise);
          auto local_all_mine = qnn_worker->computeOutVFBatch(all_states, *randomized_action);
          std::transform(all_mine->begin(), all_mine->end(), local_all_mine->begin(), all_mine->begin(), std::plus<double>());
          delete local_all_mine;
          delete randomized_action;
        }
        double factor_ = 1.f/((double)qac_sample);
        std::transform(all_mine->begin(), all_mine->end(), all_mine->begin(), std::bind1st(std::multiplies<double>(), factor_));
      }

      std::vector<double>* current_all_next_actions = all_next_actions;
      if(recompute_next_ac)
        current_all_next_actions = ann_worker->computeOutBatch(all_next_states);
      
      if(qnextac_sample <= 1){
        all_nextV = qnn_worker->computeOutVFBatch(all_next_states, *current_all_next_actions);
      } else {
        all_nextV = new std::vector<double>(traj.size(), 0.f);
        for(uint j=0;j<qnextac_sample;j++){
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*current_all_next_actions, noise);
          auto local_all_nextV = qnn_worker->computeOutVFBatch(all_next_states, *randomized_action);
          std::transform(all_nextV->begin(), all_nextV->end(), local_all_nextV->begin(), all_nextV->begin(), std::plus<double>());
          delete local_all_nextV;
          delete randomized_action;
        }
        double factor_ = 1.f/((double)qnextac_sample);
        std::transform(all_nextV->begin(), all_nextV->end(), all_nextV->begin(), std::bind1st(std::multiplies<double>(), factor_));
      }
      
      if(recompute_next_ac)
        delete current_all_next_actions;
      
      // compute delta
      i=0;
      for (auto it : traj) {
        double v_target = it.r;
        if (!it.goal_reached) {
          double nextV = all_nextV->at(i);
          v_target += this->gamma * nextV;
        }
        
        deltas[i] = v_target - all_mine->at(i);
        ++i;
      }
      
      // compute disable_back
      i=0;
      for(auto it : traj) {
        if(deltas[i] <= 0.0000000000f)
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + i * this->nb_motors);
        i++;
      }
      
      // unless deter update
      qnn->ZeroGradParameters();
      ann->ZeroGradParameters();
      
      delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
      const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      qnn->critic_backward();
      const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
      const double* critic_action_diff = critic_action_blob->cpu_diff();
      
      const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
      auto ac_diff = actor_actions_blob->mutable_cpu_diff();
      for(i=0; i<actor_actions_blob->count(); i++) {
        if(disable_back[i]) {
          ac_diff[i] = critic_action_diff[i];
        } else {
          double x = all_actions[i] - all_actions_outputs->at(i);
          ac_diff[i] = -x;
        }
      }
      
      if(inverting_grad){

        for (uint n = 0; n < traj.size(); ++n) {
          for (uint h = 0; h < nb_motors; ++h) {
            int offset = actor_actions_blob->offset(n,h,0,0);
            double diff = ac_diff[offset];
            double output = all_actions_outputs->at(offset);
            double min = -1.0; 
            double max = 1.0;
            if (diff < 0) {
              diff *= (max - output) / (max - min);
            } else if (diff > 0) {
              diff *= (output - min) / (max - min);
            }
            ac_diff[offset] = diff;
          }
        }
      }
      
      ann->actor_backward();
      ann->getSolver()->ApplyUpdate();
      ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
      
      // Soft update of targets networks
      qnn_target->soft_update(*qnn, tau_soft_update);
      ann_target->soft_update(*ann, tau_soft_update);
      
      delete q_targets;
      delete all_actions_outputs;
      delete all_next_actions;
      
      delete all_nextV;
      delete all_mine;
    }
  
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return qnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
  }

  void save(const std::string& path, bool, bool) override {
//     if(save_best && best_population.size() > 0){
//       auto it = best_population.begin();
//       bib::XMLEngine::save(it->trajectory_a, "trajectory_a", "best_trajectory_a.data");
//     } else {
      ann->save(path+".actor");
      qnn->save(path+".critic");
//     }
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
  
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint kMinibatchSize;
  uint replay_memory;
  uint force_more_update;
  uint batch_norm_actor, batch_norm_critic;
  uint actor_output_layer_type, hidden_layer_type;
  
  bool inverting_grad, bn_adapt;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::deque<std::vector<double>> last_trajectory_a;
  
  bool qac_target, aac_target, recompute_next_ac, onpolac, qmu;
  uint qac_sample, qnextac_sample;
  
  uint episode = 0;
  
  struct my_pol_dpmt{
    MLP* ann;
    MLP* qnn;
    double J;
    std::deque<std::vector<double>> trajectory_a;
    
    bool operator< (const my_pol_dpmt& b) const {
      return J > b.J;
    }
  };
  
  MLP* ann, *ann_target;
  MLP* qnn, *qnn_target;
  MLP* ann_testing;
  
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

