#ifndef NEURALFITTEDACAG_HPP
#define NEURALFITTEDACAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/filesystem.hpp>

#include "nn/MLP.hpp"
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
  bool labeled;


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
    ar& BOOST_SERIALIZATION_NVP(labeled);
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

  inline double operator[](size_t const N) const {
    return s[N];
  }

  bool same_state(const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return false;
    }

    return true;
  }

} sample;

class NeuralFittedACAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;

  NeuralFittedACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~NeuralFittedACAg() {
    delete qnn;
    delete ann;

    delete hidden_unit_q;
    delete hidden_unit_a;

    for(auto p : old_executed_policies)
      delete p;

    if(target_network) {
      delete qnn_target;
      delete ann_target;
    }

#ifdef POOL_FOR_TESTING
    for (auto i : best_pol_population)
      delete i.ann;
#endif
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool last) override {

    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {
      double p0 = 1.f;
      for(uint i=0; i < nb_motors; i++){
        p0 *= bib::Proba<double>::truncatedGaussianDensity(last_action->at(i), last_pure_action->at(i), noise);
      }

      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || last, p0, 0., false};
      if(goal_reached && reward > rmax) {
        rmax = reward;
        rmax_labeled = true;
      }
      insertSample(sa);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise) { //e-greedy
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

  void insertSample(const sample& sa) {

    if(no_forgot_offline)
      trajectory_noforgot.push_back(sa);
    if(!on_policy_update) {
      if(trajectory.size() >= replay_memory)
        trajectory.pop_front();
      trajectory.push_back(sa);

      if(force_online_update)
        update_actor_critic();

    } else {
      last_trajectory.push_back(sa);
    }
    current_trajectory.push_back(sa);
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                       = pt->get<double>("agent.noise");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    mini_batch_size             = pt->get<uint>("agent.mini_batch_size");
    replay_memory               = pt->get<uint>("agent.replay_memory");
    reset_qnn                   = pt->get<bool>("agent.reset_qnn");
    force_online_update         = pt->get<bool>("agent.force_online_update");
    nb_actor_updates            = pt->get<uint>("agent.nb_actor_updates");
    nb_critic_updates           = pt->get<uint>("agent.nb_critic_updates");
    nb_fitted_updates           = pt->get<uint>("agent.nb_fitted_updates");
    nb_internal_critic_updates  = pt->get<uint>("agent.nb_internal_critic_updates");
    alpha_a                     = pt->get<double>("agent.alpha_a");
    alpha_v                     = pt->get<double>("agent.alpha_v");
    batch_norm                  = pt->get<uint>("agent.batch_norm");
    max_stabilizer              = pt->get<bool>("agent.max_stabilizer");
    min_stabilizer              = pt->get<bool>("agent.min_stabilizer");
    minibatcher                 = pt->get<uint>("agent.minibatcher");
    sampling_strategy           = pt->get<uint>("agent.sampling_strategy");
    fishing_policy              = pt->get<uint>("agent.fishing_policy");
    inverting_grad              = pt->get<bool>("agent.inverting_grad");
    decay_v                     = pt->get<double>("agent.decay_v");
    target_network              = pt->get<bool>("agent.target_network");
    tau_soft_update             = pt->get<double>("agent.tau_soft_update");
    weighting_strategy          = pt->get<uint>("agent.weighting_strategy");
    last_layer_actor            = pt->get<uint>("agent.last_layer_actor");
    reset_ann                   = pt->get<bool>("agent.reset_ann");
    no_forgot_offline           = pt->get<bool>("agent.no_forgot_offline");
    mixed_sampling              = pt->get<bool>("agent.mixed_sampling");
    hidden_layer_type           = pt->get<uint>("agent.hidden_layer_type");
    stop_reset                  = pt->get<bool>("agent.stop_reset");
    only_one_traj               = pt->get<bool>("agent.only_one_traj");
    only_one_traj_actor         = pt->get<bool>("agent.only_one_traj_actor");

    on_policy_update            = max_stabilizer;
    rmax_labeled                = false;
    rmax                        = std::numeric_limits<double>::lowest();

    old_executed_policies.clear();

#ifdef CAFFE_CPU_ONLY
    LOG_INFO("CPU mode");
    (void) command_args;
#else
    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0) {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
      LOG_INFO("CPU mode");
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
      caffe::Caffe::SetDevice(0);
      LOG_INFO("GPU mode");
    }
#endif

    if(reset_qnn && minibatcher != 0) {
      LOG_INFO("option splash -> cannot reset_qnn and count on minibatch to train a new Q function");
      exit(1);
    }

    if(minibatcher == 0 && sampling_strategy > 0 && !no_forgot_offline) {
      LOG_INFO("option splash -> cannot have a sampling stat without sampling");
      exit(1);
    }
    
    if(mixed_sampling && !no_forgot_offline){
      LOG_INFO("option splash -> cannot have mixed_sampling stat without no_forgot_offline");
      exit(1);
    }

    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                  alpha_v,
                  mini_batch_size,
                  decay_v,
                  hidden_layer_type, batch_norm,
                  weighting_strategy > 0);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor, batch_norm);

    if(target_network) {
      qnn_target = new MLP(*qnn, false);
      ann_target = new MLP(*ann, false);
    }
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    learning = _learning;
    
    if(only_one_traj)
      trajectory.clear();
    
    current_trajectory.clear();
  }

  void computePTheta(const std::deque< sample >& vtraj, double *ptheta) {

    uint number_minibatch = (int)(trajectory.size() / mini_batch_size) + 1;
    auto it1 = vtraj.cbegin();
    auto it2 = vtraj.cbegin();
    uint index = 0;

    for(uint n=0; n < number_minibatch; n++) {
      std::vector<double> all_states(mini_batch_size * nb_sensors, 0.f);

      for (uint i=0; i<mini_batch_size && it1 != vtraj.cend(); i++) {
        std::copy(it1->s.begin(), it1->s.end(), all_states.begin() + i * nb_sensors);
        ++it1;
      }

      auto all_next_actions = ann->computeOutBatch(all_states);

      for(uint i=0; i<mini_batch_size && it2 != vtraj.cend(); i++) {
        sample sm = *it2;
        double p0 = 1.f;
        for(uint j=0; j < nb_motors; j++){
            p0 *= bib::Proba<double>::truncatedGaussianDensity(sm.a[j], all_next_actions->at(i*nb_motors+j), noise);
        }

        ptheta[index] = p0;
        index++;
        ++it2;
      }

      delete all_next_actions;
    }

    ASSERT(index == vtraj.size(), "");
  }
  
  void computePThetaBatch(const std::deque< sample >& vtraj, double *ptheta, const std::vector<double>* all_next_actions){
    uint i=0;
    for(auto it : vtraj) {
      double p0 = 1.f;
      for(uint j=0; j < nb_motors; j++){
        p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_next_actions->at(i*nb_motors+j), noise);
      }
      
      ptheta[i] = p0;
      i++;
    }
  }

  double sum_QSA(const std::deque<sample>& vtraj, MLP* policy) {
    //vtraj should already be the size of mini_batch_size
    std::vector<double> all_states(mini_batch_size * nb_sensors);

    auto it1 = vtraj.cbegin();
    for (uint i=0; i<mini_batch_size; i++) {
      std::copy(it1->s.begin(), it1->s.end(), all_states.begin() + i * nb_sensors);
      ++it1;
    }
    auto all_actions_outputs = policy->computeOutBatch(all_states);

    auto all_qsa = qnn->computeOutVFBatch(all_states, *all_actions_outputs);

    double sum = std::accumulate(all_qsa->cbegin(), all_qsa->cend(), (double) 0.f);
    
    delete all_qsa;
    delete all_actions_outputs;

    return sum /((double) vtraj.size());
  }

  void sample_transition(std::deque<sample>& traj, const std::deque<sample>& from, uint nb_sample) {
    if(sampling_strategy <= 1) {
      for(uint i=0; i<nb_sample; i++) {
        int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
        traj[i] = from[r];
      }
    } else if(sampling_strategy > 1) {
      std::vector<double> weights(from.size());
      double* ptheta = new double[from.size()];
      computePTheta(from, ptheta);

      uint i=0;
      if(sampling_strategy == 2)
        for(auto it = from.cbegin(); it != from.cend() ; ++it) {
          weights[i] = ptheta[i] / it->p0;
          i++;
        }
      else //sampling_strategy = 3
        for(auto it = from.cbegin(); it != from.cend() ; ++it) {
          weights[i] = std::min( (double) 1.0f, ptheta[i] / it->p0);
          i++;
        }
      delete[] ptheta;

      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      for(i = 0; i < nb_sample; i++)
        traj[i] = from[dist(*bib::Seed::random_engine())];
    }
  }

  void label_onpoltarget() {
    CHECK_GT(last_trajectory.size(), 0) << "Need at least one transition to label.";

    sample& last = last_trajectory[last_trajectory.size()-1];
    last.onpolicy_target = last.r;// Q-Val is just the final reward
    last.labeled=true;
    for (int i=last_trajectory.size()-2; i>=0; --i) {
      sample& t = last_trajectory[i];
      float reward = t.r;
      float target = last_trajectory[i+1].onpolicy_target;
      t.onpolicy_target = reward + gamma * target;
      t.labeled = true;
    }

  }

  void critic_update(uint iter) {
    std::deque<sample>* traj = &trajectory;
    uint number_of_run = std::max((uint) 1, minibatcher * iter);

    for(uint batch_sampling=0; batch_sampling < number_of_run; batch_sampling++) { //at least one time
      if(minibatcher != 0) {
        std::deque<sample>* sub_traj = new std::deque<sample>(mini_batch_size);
        sample_transition(*sub_traj, trajectory, mini_batch_size);
        traj = sub_traj;
      }

      //compute \pi(s_{t+1})
      std::vector<double> all_next_states(traj->size() * nb_sensors);
      std::vector<double> all_states(traj->size() * nb_sensors);
      std::vector<double> all_actions(traj->size() * nb_motors);
      uint i=0;
      for (auto it : *traj) {
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
        std::copy(it.a.begin(), it.a.end(), all_actions.begin() + i * nb_motors);
        i++;
      }
      
      std::vector<double>* all_next_actions;
      if(target_network)
        all_next_actions = ann_target->computeOutBatch(all_next_states);
      else
        all_next_actions = ann->computeOutBatch(all_next_states);
      
      //compute next q
      std::vector<double>* q_targets;
      std::vector<double>* q_targets_weights = nullptr;
      double* ptheta = nullptr;
      if(target_network)
        q_targets = qnn_target->computeOutVFBatch(all_next_states, *all_next_actions);
      else
        q_targets = qnn->computeOutVFBatch(all_next_states, *all_next_actions);

      if(weighting_strategy != 0) {
        q_targets_weights = new std::vector<double>(q_targets->size(), 1.0f);
        if(weighting_strategy > 1) {
          ptheta = new double[traj->size()];
//           computePTheta(*traj, ptheta);
          computePThetaBatch(*traj, ptheta, all_next_actions);
        }
      }
      delete all_next_actions;

      //adjust q_targets
      i=0;
      for (auto it : *traj) {
        if(it.goal_reached)
          q_targets->at(i) = it.r;
        else {
          q_targets->at(i) = it.r + gamma * q_targets->at(i);

          if(max_stabilizer && it.labeled && q_targets->at(i) < it.onpolicy_target )
            q_targets->at(i) = it.onpolicy_target;

          if(min_stabilizer && rmax_labeled && q_targets->at(i) > rmax)
            q_targets->at(i) = rmax;
        }

        if(weighting_strategy==1)
          q_targets_weights->at(i)=1.0f/it.p0;
        else if(weighting_strategy==2)
          q_targets_weights->at(i)=ptheta[i]/it.p0;
        else if(weighting_strategy==3)
          q_targets_weights->at(i)=std::min((double)1.0f, ptheta[i]/it.p0);

        i++;
      }

      if(reset_qnn && (stop_reset ? episode < 1000 : true)) {
        delete qnn;
        qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                      alpha_v,
                      mini_batch_size,
                      decay_v,
                      hidden_layer_type, batch_norm,
                      weighting_strategy > 0);
      }

      //Update critic
      qnn->InputDataIntoLayers(all_states.data(), all_actions.data(), q_targets->data());
      if(weighting_strategy != 0)
        qnn->setWeightedSampleVector(q_targets_weights->data());
      if(minibatcher != 0)
        qnn->getSolver()->Step(1);
      else
        qnn->getSolver()->Step(iter);

      delete q_targets;
      if(weighting_strategy != 0) {
        delete q_targets_weights;
        if(weighting_strategy > 1)
          delete[] ptheta;
      }

      //usefull?
      qnn->ZeroGradParameters();

      if(minibatcher != 0)
        delete traj;

      if(target_network) {
        qnn_target->soft_update(*qnn, tau_soft_update);
      }
    }
  }

  void actor_update_grad() {
    std::deque<sample>* traj = &trajectory;
    if(only_one_traj_actor)
      traj = &current_trajectory;
    uint number_of_run = std::max((uint) 1, minibatcher);

    for(uint batch_sampling=0; batch_sampling < number_of_run; batch_sampling++) { //at least one time
      if(minibatcher != 0) {
        std::deque<sample>* sub_traj = new std::deque<sample>(mini_batch_size);
        sample_transition(*sub_traj, trajectory, mini_batch_size);
        traj = sub_traj;
      }

      std::vector<double> all_states(traj->size() * nb_sensors);
      uint i=0;
      for (auto it : *traj) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
        i++;
      }

      //Update actor
      qnn->ZeroGradParameters();
      ann->ZeroGradParameters();

      auto all_actions_outputs = ann->computeOutBatch(all_states);
//       shrink_actions(all_actions_outputs);

      delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);

      const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : *traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      qnn->getNN()->BackwardFrom(qnn->GetLayerIndex(MLP::q_values_layer_name));
      const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
      if(inverting_grad) {
        double* action_diff = critic_action_blob->mutable_cpu_diff();

        for (uint n = 0; n < traj->size(); ++n) {
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

      delete all_actions_outputs;

      if(minibatcher != 0)
        delete traj;

      if(target_network) {
        ann_target->soft_update(*ann, tau_soft_update);
      }
    }
  }

  void update_actor_critic() {
    if(!learning)
      return;

    if(minibatcher == 0 && trajectory.size() != mini_batch_size) {
      mini_batch_size = trajectory.size();
      qnn->increase_batchsize(mini_batch_size);
      ann->increase_batchsize(mini_batch_size);
      if(target_network){
        qnn_target->increase_batchsize(mini_batch_size);
        ann_target->increase_batchsize(mini_batch_size);
      }
    }
    
    if(no_forgot_offline && trajectory_noforgot.size() > trajectory.size() 
      && trajectory_noforgot.size() > replay_memory){
      trajectory.resize(replay_memory);
      sample_transition(trajectory, trajectory_noforgot, replay_memory);
    }

    std::vector<MLP*> candidate_policies(nb_fitted_updates);
    std::vector<double> candidate_policies_scores(nb_fitted_updates);
    std::deque<sample>* samples_for_score = nullptr;

    if(fishing_policy >= 1) {
      samples_for_score = new std::deque<sample>(mini_batch_size);
      sample_transition(*samples_for_score, trajectory, mini_batch_size);
    }

    for(uint n=0; n<nb_fitted_updates; n++) {
      for(uint i=0; i<nb_critic_updates ; i++)
        critic_update(nb_internal_critic_updates);

      if(fishing_policy > 0) {
        candidate_policies[n]=new MLP(*ann, true);

        if(fishing_policy == 1) {
          candidate_policies_scores[n]=sum_QSA(*samples_for_score, candidate_policies[n]);
        } else if(fishing_policy == 2) {
          candidate_policies_scores[n]= - qnn->error();
        }
      }
      
      if(reset_ann && (stop_reset ? episode < 1000 : true)) {
        delete ann;
        ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor, batch_norm);
      }

      for(uint i=0; i<nb_actor_updates ; i++)
        actor_update_grad();
      
      if(no_forgot_offline && trajectory_noforgot.size() > trajectory.size() 
        && trajectory_noforgot.size() > replay_memory && mixed_sampling){
        trajectory.resize(replay_memory);
        sample_transition(trajectory, trajectory_noforgot, replay_memory);
      }
    }

    if(fishing_policy > 0) {
      if(fishing_policy <= 2) {
        double mmax = *std::max_element(candidate_policies_scores.begin(), candidate_policies_scores.end());
        uint index_best = 0;
        while(candidate_policies_scores[index_best] < mmax)
          index_best++;

        ASSERT(index_best < nb_fitted_updates, "out of range");
        delete ann;
        ann = new MLP(*candidate_policies[index_best], true);

        delete samples_for_score;
      }

      for(auto pol : candidate_policies)
        delete pol;
    }
  }

  void end_episode() override {
    episode++;
    
    if(fishing_policy > 0)
      old_executed_policies.push_back(new MLP(*ann, true));

    if(on_policy_update) {
      while(trajectory.size() + last_trajectory.size() > replay_memory)
        trajectory.pop_front();

      label_onpoltarget();
      auto it = trajectory.end();
      trajectory.insert(it, last_trajectory.begin(), last_trajectory.end());
      last_trajectory.clear();
    }

    update_actor_critic();
  }
#ifdef POOL_FOR_TESTING
  void start_instance(bool learning) override {
    if(!learning && best_pol_population.size() > 0) {
      to_be_restaured_ann = ann;
      auto it = best_pol_population.begin();
      ++it;
      ann = best_pol_population.begin()->ann;
    } else if(learning) {
      to_be_restaured_ann = new MLP(*ann, false);
    }
  }

  void end_instance(bool learning) override {
    if(!learning && best_pol_population.size() > 0) {
      //restore ann
      ann = to_be_restaured_ann;
    } else if(learning) {
      //not totaly stable because J(the policy stored here ) != sum_weighted_reward (online updates)

      //policies pool for testing
      if(best_pol_population.size() == 0 || best_pol_population.rbegin()->J < sum_weighted_reward) {
        if(best_pol_population.size() > 10) {
          //remove smallest
          auto it = best_pol_population.end();
          --it;
          delete it->ann;
          best_pol_population.erase(it);
        }

        MLP* pol_fitted_sample=to_be_restaured_ann;
        best_pol_population.insert({pol_fitted_sample,sum_weighted_reward, 0});
      } else
        delete to_be_restaured_ann;

    }
  }
#endif
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
    return qnn->computeOutVF(perceptions, actions);
  }

  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN :
                                 arch::policy_type::GREEDY,
                                 noise, decision_each);
  }

  void save(const std::string& path, bool) override {
    ann->save(path+".actor");
    qnn->save(path+".critic");
//      bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    qnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward
#ifndef NDEBUG
        << " " << std::setw(8) << std::fixed << std::setprecision(5) << noise
        << " " << trajectory.size()
        << " " << ann->weight_l1_norm()
        << " " << std::fixed << std::setprecision(7) << qnn->error()
        << " " << qnn->weight_l1_norm()
#endif
        ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << trajectory.size() ;
  }


 private:
  uint nb_sensors;
  uint episode = 0;

  double noise;
  double rmax;
  bool rmax_labeled;
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint mini_batch_size;
  uint replay_memory, nb_actor_updates, nb_critic_updates, nb_fitted_updates, nb_internal_critic_updates;
  bool stop_reset, only_one_traj, only_one_traj_actor;
  double alpha_a;
  double alpha_v;
  double decay_v;
  double tau_soft_update;

  uint batch_norm, minibatcher, sampling_strategy, fishing_policy, weighting_strategy, last_layer_actor, hidden_layer_type;
  bool learning, on_policy_update, reset_qnn, force_online_update, max_stabilizer, min_stabilizer, inverting_grad;
  bool target_network, reset_ann, no_forgot_offline, mixed_sampling;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::vector<sample> last_trajectory;
  std::deque<sample> trajectory_noforgot;
  std::deque<sample> current_trajectory;

  std::list<MLP*> old_executed_policies;

  MLP* ann;
  MLP* qnn;

  MLP* ann_target;
  MLP* qnn_target;

#ifdef POOL_FOR_TESTING
  struct my_pol {
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
};

#endif

