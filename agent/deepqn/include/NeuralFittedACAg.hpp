#ifndef NEURALFITTEDACAG_HPP
#define NEURALFITTEDACAG_HPP

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

class NeuralFittedACAg : public arch::AACAgent<MLP, AgentGPUProgOptions> {
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
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool last) override {

    vector<double>* next_action = ann->computeOut(sensors);
//     shrink_actions(next_action);

    if (last_action.get() != nullptr && learning) {
      double p0 = 1.f;
      for(uint i=0; i < nb_motors; i++)
        p0 *=
          
exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));

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
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
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

    if(!on_policy_update) {
      if(trajectory.size() >= replay_memory)
        trajectory.pop_front();
      trajectory.push_back(sa);

      if(force_online_update)
        update_actor_critic();

    } else {
      last_trajectory.push_back(sa);
    }
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
    double decay_v              = pt->get<double>("agent.decay_v");

    on_policy_update            = max_stabilizer;
    rmax_labeled                = false;
    rmax                        = std::numeric_limits<double>::lowest();

    old_executed_policies.clear();

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

    if(reset_qnn && minibatcher != 0) {
      LOG_DEBUG("option splash -> cannot reset_qnn and count on minibatch to train a new Q function");
      exit(1);
    }

    if(minibatcher == 0 && sampling_strategy > 0) {
      LOG_DEBUG("option splash -> cannot have a sampling stat without sampling");
      exit(1);
    }

    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, mini_batch_size, decay_v, batch_norm);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, !inverting_grad, batch_norm);
  }

  void shrink_actions(vector<double>* next_action) {
    for(uint i=0; i < nb_motors ; i++)
      if(next_action->at(i) > 1.f)
        next_action->at(i)=1.f;
      else if(next_action->at(i) < -1.f)
        next_action->at(i)=-1.f;
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    learning = _learning;
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
//       shrink_actions(all_next_actions);

      for(uint i=0; i<mini_batch_size && it2 != vtraj.cend(); i++) {
        sample sm = *it2;
        double p0 = 1.f;
        for(uint j=0; i < nb_motors; i++)
          p0 *= exp(-(all_next_actions->at(i*nb_motors+j)-sm.a[j])*(all_next_actions->at(i*nb_motors+j)-sm.a[j])/
                    (2.f*noise*noise));

        ptheta[index] = p0;
        index++;
        ++it2;
      }

      delete all_next_actions;
    }

    ASSERT(index == vtraj.size(), "");
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
//     shrink_actions(all_actions_outputs); //sure_shrink to false from DDPG

    auto all_qsa = qnn->computeOutVFBatch(all_states, *all_actions_outputs);

    delete all_actions_outputs;
    delete all_qsa;

    double sum = std::accumulate(all_qsa->cbegin(), all_qsa->cend(), (double) 0.f);

    return sum /((double) vtraj.size());
  }

  void sample_transition(std::deque<sample>& traj, const std::deque<sample>& from, uint nb_sample) {
    if(sampling_strategy == 1) {
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

      auto all_next_actions = ann->computeOutBatch(all_next_states);
//       shrink_actions(all_next_actions);
      //compute next q
      auto q_targets = qnn->computeOutVFBatch(all_next_states, *all_next_actions);
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

        i++;
      }

      if(reset_qnn) {
        delete qnn;
        qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, alpha_v, mini_batch_size, -1, batch_norm);
      }

      //Update critic
      qnn->InputDataIntoLayers(all_states.data(), all_actions.data(), q_targets->data());
      if(minibatcher != 0)
        qnn->getSolver()->Step(1);
      else
        qnn->getSolver()->Step(iter);

      delete q_targets;

      //usefull?
      qnn->ZeroGradParameters();

      if(minibatcher != 0)
        delete traj;
    }
  }

  void actor_update_grad() {
    std::deque<sample>* traj = &trajectory;
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
      //shrink_actions(all_actions_outputs); //sure_shrink to false from DDPG

      delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);

      const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : *traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      qnn->getNN()->BackwardFrom(qnn->GetLayerIndex(MLP::q_values_layer_name));
      const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
      if(inverting_grad){
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
    }
  }

  void update_actor_critic() {
    if(!learning || trajectory.size() < mini_batch_size)
      return;

    if(minibatcher == 0 && trajectory.size() != mini_batch_size) {
      mini_batch_size = trajectory.size();
      qnn->increase_batchsize(mini_batch_size);
      ann->increase_batchsize(mini_batch_size);
    }

    std::vector<MLP*> candidate_policies(nb_fitted_updates);
    std::vector<double> candidate_policies_scores(nb_fitted_updates);
    std::deque<sample>* samples_for_score = nullptr;
    
    if(fishing_policy == 1){
      samples_for_score = new std::deque<sample>(mini_batch_size);
      sample_transition(*samples_for_score, trajectory, mini_batch_size);
    }

    for(uint n=0; n<nb_fitted_updates; n++) {
      for(uint i=0; i<nb_critic_updates ; i++)
        critic_update(nb_internal_critic_updates);

      if(fishing_policy > 0) {
        candidate_policies[n]=new MLP(*ann, true);

        if(fishing_policy == 0) {
          candidate_policies_scores[n]=sum_QSA(*samples_for_score, *candidate_policies.crbegin());
        }
      }

      for(uint i=0; i<nb_actor_updates ; i++)
        actor_update_grad();
    }

    if(fishing_policy > 0) {
      if(fishing_policy == 1) {
        double mmax = *std::max_element(candidate_policies_scores.begin(), candidate_policies_scores.end());
        uint index_best = 0;
        while(candidate_policies_scores[index_best] < mmax)
          index_best++;
        
        delete ann;
        ann = new MLP(*candidate_policies[index_best], true);
        
        delete samples_for_score;
      }

      for(auto pol : candidate_policies)
        delete pol;
    }
  }

  void end_episode() override {
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

  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
    return qnn->computeOutVF(perceptions, actions);
  }

  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return new arch::Policy<MLP>(new MLP(*ann, true) , gaussian_policy ? arch::policy_type::GAUSSIAN :
                                 arch::policy_type::GREEDY,
                                 noise, decision_each);
  }

  void save(const std::string& path) override {
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(8) <<
        std::fixed << std::setprecision(5) << noise << " " << trajectory.size() << " " << ann->weight_l1_norm() << " "
        << std::fixed << std::setprecision(7) << qnn->error() << " " << qnn->weight_l1_norm();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << trajectory.size() ;
  }


 private:
  uint nb_sensors;

  double noise;
  double rmax;
  bool rmax_labeled;
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint mini_batch_size;
  uint replay_memory, nb_actor_updates, nb_critic_updates, nb_fitted_updates, nb_internal_critic_updates;
  double alpha_a;
  double alpha_v;

  uint batch_norm, minibatcher, sampling_strategy, fishing_policy;
  bool learning, on_policy_update, reset_qnn, force_online_update, max_stabilizer, min_stabilizer, inverting_grad;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::vector<sample> last_trajectory;

  std::list<MLP*> old_executed_policies;

  MLP* ann;
  MLP* qnn;
};

#endif

