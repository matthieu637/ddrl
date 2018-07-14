#ifndef NEURALFITTEDACAG_HPP
#define NEURALFITTEDACAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

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
#warning POOL_FOR_TESTING not implemented
#undef POOL_FOR_TESTING
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

class NeuralFittedMultiACAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;

  NeuralFittedMultiACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~NeuralFittedMultiACAg() {
    for(auto i : qnn)
      delete i;
    for(auto i : ann)
      delete i;
    
    if(policy_selection == 2 || policy_selection == 12 || policy_selection == 4)
      delete global_qnn;

    delete hidden_unit_q;
    delete hidden_unit_a;

#ifdef POOL_FOR_TESTING
    for (auto i : best_pol_population)
      delete i.ann;
#endif
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool last) override {

    vector<double>* next_action = nullptr;
    if(policy_selection == 0) {
      next_action = ann[0]->computeOut(sensors);
      double bestS = qnn[0]->computeOutVF(sensors, *next_action);
      uint best_index = 0;
      delete next_action;

      for(uint mm=1; mm < multi_policies; mm++) {
        next_action = ann[mm]->computeOut(sensors);
        double lscore = qnn[mm]->computeOutVF(sensors, *next_action);
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
        delete next_action;
      }
      next_action = ann[best_index]->computeOut(sensors);
    } else if(policy_selection == 1) {
      next_action = ann[(uint)(bib::Utils::rand01()*multi_policies)]->computeOut(sensors);
    } else if(policy_selection == 2) {
      next_action = ann[0]->computeOut(sensors);
      double bestS = global_qnn->computeOutVF(sensors, *next_action);
      uint best_index = 0;
      delete next_action;
      
      for(uint mm=1; mm < multi_policies; mm++) {
        next_action = ann[mm]->computeOut(sensors);
        double lscore = global_qnn->computeOutVF(sensors, *next_action);
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
        delete next_action;
      }
      next_action = ann[best_index]->computeOut(sensors);
    } else if(policy_selection == 3) {
      for(uint mm=0; mm < multi_policies; mm++){
        next_action = ann[mm]->computeOut(sensors);
        sum_onlineQSA[mm] += qnn[mm]->computeOutVF(sensors, *next_action);
        delete next_action;
      }
      
      double bestS = sum_onlineQSA[0];
      uint best_index = 0;
      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = sum_onlineQSA[mm];
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      next_action = ann[best_index]->computeOut(sensors);
    } else if(policy_selection == 4) {
      for(uint mm=0; mm < multi_policies; mm++){
        next_action = ann[mm]->computeOut(sensors);
        sum_onlineQSA[mm] += global_qnn->computeOutVF(sensors, *next_action);
        delete next_action;
      }
      
      double bestS = sum_onlineQSA[0];
      uint best_index = 0;
      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = sum_onlineQSA[mm];
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      next_action = ann[best_index]->computeOut(sensors);
    } else if(policy_selection == 5) {
      for(uint mm=0; mm < multi_policies; mm++){
        next_action = ann[mm]->computeOut(sensors);
        sum_onlineQSA[mm] = qnn[mm]->computeOutVF(sensors, *next_action) + gamma * sum_onlineQSA[mm];
        delete next_action;
      }
      
      double bestS = sum_onlineQSA[0];
      uint best_index = 0;
      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = sum_onlineQSA[mm];
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      next_action = ann[best_index]->computeOut(sensors);
    } else if(policy_selection >= 10) {
      next_action = ann[policy_selected]->computeOut(sensors);
    }


    if (last_action.get() != nullptr && learning) {
      double p0 = 1.f;
      for(uint i=0; i < nb_motors; i++) {
        p0 *= bib::Proba<double>::truncatedGaussianDensity(last_action->at(i), last_pure_action->at(i), noise);
      }

      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || last, p0, 0., false};
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

    if(trajectory.size() >= replay_memory)
      trajectory.pop_front();
    trajectory.push_back(sa);

  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                       = pt->get<double>("agent.noise");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    mini_batch_size             = pt->get<uint>("agent.mini_batch_size");
    replay_memory               = pt->get<uint>("agent.replay_memory");
    reset_qnn                   = pt->get<bool>("agent.reset_qnn");
    nb_actor_updates            = pt->get<uint>("agent.nb_actor_updates");
    nb_critic_updates           = pt->get<uint>("agent.nb_critic_updates");
    nb_fitted_updates           = pt->get<uint>("agent.nb_fitted_updates");
    nb_internal_critic_updates  = pt->get<uint>("agent.nb_internal_critic_updates");
    alpha_a                     = pt->get<double>("agent.alpha_a");
    alpha_v                     = pt->get<double>("agent.alpha_v");
    batch_norm                  = pt->get<uint>("agent.batch_norm");
    sampling_strategy           = pt->get<uint>("agent.sampling_strategy");
    inverting_grad              = pt->get<bool>("agent.inverting_grad");
    decay_v                     = pt->get<double>("agent.decay_v");
    weighting_strategy          = pt->get<uint>("agent.weighting_strategy");
    last_layer_actor            = pt->get<uint>("agent.last_layer_actor");
    reset_ann                   = pt->get<bool>("agent.reset_ann");
    no_forgot_offline           = pt->get<bool>("agent.no_forgot_offline");
//     mixed_sampling              = pt->get<bool>("agent.mixed_sampling");
    hidden_layer_type           = pt->get<uint>("agent.hidden_layer_type");
    multi_policies              = pt->get<uint>("agent.multi_policies");
    force_same_data_mp          = pt->get<bool>("agent.force_same_data_mp");
    policy_selection            = pt->get<uint>("agent.policy_selection");

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
    if(sampling_strategy > 0 && !no_forgot_offline) {
      LOG_INFO("option splash -> cannot have a sampling stat without sampling");
      exit(1);
    }

//     if(mixed_sampling && !no_forgot_offline) {
//       LOG_INFO("option splash -> cannot have mixed_sampling stat without no_forgot_offline");
//       exit(1);
//     }

    if(!force_same_data_mp && !no_forgot_offline) {
      LOG_INFO("option splash -> cannot have different data without keeping every datas");
      exit(1);
    }

    qnn.resize(multi_policies);
    ann.resize(multi_policies);
    sum_onlineQSA.resize(multi_policies);

    for(uint mm=0; mm < multi_policies; mm++) {
      qnn[mm] = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                        alpha_v,
                        mini_batch_size,
                        decay_v,
                        hidden_layer_type, batch_norm,
                        weighting_strategy > 0);

      ann[mm] = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor,
                        batch_norm);
    }
    
    if(policy_selection == 2 || policy_selection == 12 || policy_selection == 4)
    global_qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                         alpha_v,
                         mini_batch_size,
                         decay_v,
                         hidden_layer_type, batch_norm,
                         weighting_strategy > 0);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    learning = _learning;
    
    
    for(uint mm=0; mm < multi_policies; mm++)
      sum_onlineQSA[mm] = 0.f;

    if(policy_selection == 10 && trajectory.size() > 0 && multi_policies > 1) {
      double bestS = sum_QSA(trajectory, ann[0], qnn[0]);
      uint best_index = 0;

      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = sum_QSA(trajectory, ann[mm], qnn[mm]);
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      policy_selected = best_index;
    } else if(policy_selection == 11 && trajectory.size() > 0 && multi_policies > 1){
      double bestS = qnn[0]->error();
      uint best_index = 0;
      
      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = qnn[mm]->error();
        if(lscore < bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      policy_selected = best_index;
    } else if(policy_selection == 12 && trajectory.size() > 0 && multi_policies > 1){
      double bestS = sum_QSA(trajectory, ann[0], global_qnn);
      uint best_index = 0;
      
      for(uint mm=1; mm < multi_policies; mm++) {
        double lscore = sum_QSA(trajectory, ann[mm], global_qnn);
        if(lscore > bestS) {
          bestS = lscore;
          best_index = mm;
        }
      }
      policy_selected = best_index;
    } else {
      policy_selected = (uint)(bib::Utils::rand01()*multi_policies);
    }
  }

  void computePTheta(const std::deque< sample >& vtraj, double *ptheta, MLP* _ann) {

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

      auto all_next_actions = _ann->computeOutBatch(all_states);

      for(uint i=0; i<mini_batch_size && it2 != vtraj.cend(); i++) {
        sample sm = *it2;
        double p0 = 1.f;
        for(uint j=0; j < nb_motors; j++) {
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

  void computePThetaBatch(const std::deque< sample >& vtraj, double *ptheta,
                          const std::vector<double>* all_next_actions) {
    uint i=0;
    for(auto it : vtraj) {
      double p0 = 1.f;
      for(uint j=0; j < nb_motors; j++) {
        p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_next_actions->at(i*nb_motors+j), noise);
      }

      ptheta[i] = p0;
      i++;
    }
  }

  double sum_QSA(const std::deque<sample>& vtraj, MLP* policy, MLP* _qnn) {
    //vtraj should already be the size of mini_batch_size
    std::vector<double> all_states(vtraj.size() * nb_sensors);

    auto it1 = vtraj.cbegin();
    for (uint i=0; i<vtraj.size(); i++) {
      std::copy(it1->s.begin(), it1->s.end(), all_states.begin() + i * nb_sensors);
      ++it1;
    }
    auto all_actions_outputs = policy->computeOutBatch(all_states);

    auto all_qsa = _qnn->computeOutVFBatch(all_states, *all_actions_outputs);

    double sum = std::accumulate(all_qsa->cbegin(), all_qsa->cend(), (double) 0.f);

    delete all_qsa;
    delete all_actions_outputs;

    return sum /((double) vtraj.size());
  }

  void sample_transition(std::deque<sample>& traj, const std::deque<sample>& from, uint nb_sample, MLP* _ann) {
    if(sampling_strategy <= 1) {
      for(uint i=0; i<nb_sample; i++) {
        int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
        traj[i] = from[r];
      }
    } else if(sampling_strategy > 1) {
      std::vector<double> weights(from.size());
      double* ptheta = new double[from.size()];
      ASSERT(_ann != nullptr , "wrong call");
      computePTheta(from, ptheta, _ann);

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

  void critic_update(uint iter, MLP* _qnn, MLP* _ann) {
    std::deque<sample>* traj = &trajectory;
    uint number_of_run = 1;

    for(uint batch_sampling=0; batch_sampling < number_of_run; batch_sampling++) { //at least one time
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
      all_next_actions = _ann->computeOutBatch(all_next_states);

      //compute next q
      std::vector<double>* q_targets;
      std::vector<double>* q_targets_weights = nullptr;
      double* ptheta = nullptr;
      q_targets = _qnn->computeOutVFBatch(all_next_states, *all_next_actions);

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
        }

        if(weighting_strategy==1)
          q_targets_weights->at(i)=1.0f/it.p0;
        else if(weighting_strategy==2)
          q_targets_weights->at(i)=ptheta[i]/it.p0;
        else if(weighting_strategy==3)
          q_targets_weights->at(i)=std::min((double)1.0f, ptheta[i]/it.p0);

        i++;
      }

      if(reset_qnn) {
        delete _qnn;
        _qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                       alpha_v,
                       mini_batch_size,
                       decay_v,
                       hidden_layer_type, batch_norm,
                       weighting_strategy > 0);
      }

      //Update critic
      if(weighting_strategy != 0)
        _qnn->stepCritic(all_states, all_actions, *q_targets, iter, q_targets_weights);
      else
        _qnn->stepCritic(all_states, all_actions, *q_targets, iter);

      delete q_targets;
      if(weighting_strategy != 0) {
        delete q_targets_weights;
        if(weighting_strategy > 1)
          delete[] ptheta;
      }

      //usefull?
      _qnn->ZeroGradParameters();
    }
  }

  void actor_update_grad(MLP* _qnn, MLP* _ann) {
    std::deque<sample>* traj = &trajectory;
    uint number_of_run = 1;

    for(uint batch_sampling=0; batch_sampling < number_of_run; batch_sampling++) { //at least one time
      std::vector<double> all_states(traj->size() * nb_sensors);
      uint i=0;
      for (auto it : *traj) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
        i++;
      }

      //Update actor
      _qnn->ZeroGradParameters();
      _ann->ZeroGradParameters();

      auto all_actions_outputs = _ann->computeOutBatch(all_states);
      //       shrink_actions(all_actions_outputs);

      delete _qnn->computeOutVFBatch(all_states, *all_actions_outputs);

      const auto q_values_blob = _qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      i=0;
      for (auto it : *traj)
        q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
      _qnn->critic_backward();
      const auto critic_action_blob = _qnn->getNN()->blob_by_name(MLP::actions_blob_name);
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
      const auto actor_actions_blob = _ann->getNN()->blob_by_name(MLP::actions_blob_name);
      actor_actions_blob->ShareDiff(*critic_action_blob);
      _ann->actor_backward();
      _ann->getSolver()->ApplyUpdate();
      _ann->getSolver()->set_iter(_ann->getSolver()->iter() + 1);

      delete all_actions_outputs;
    }
  }
  
  void critic_global_update(uint iter, MLP* _qnn, std::vector<MLP*>& _ann) {
    std::deque<sample>* traj = &trajectory;
    uint number_of_run = 1;
    
    for(uint batch_sampling=0; batch_sampling < number_of_run; batch_sampling++) { //at least one time
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
      
      std::vector<std::vector<double>*> all_next_actions(_ann.size());
      std::vector< std::vector<double>*> q_targets(_ann.size());
      std::vector<std::vector<double>*> q_targets_weights(_ann.size());
      std::vector<double*> ptheta(_ann.size());
      for(uint p=0;p < _ann.size(); p++){
        all_next_actions[p] = _ann[p]->computeOutBatch(all_next_states);
      
        //compute next q

        
        q_targets[p] = _qnn->computeOutVFBatch(all_next_states, *all_next_actions[p]);
        
        if(weighting_strategy != 0) {
          q_targets_weights[p] = new std::vector<double>(traj->size(), 1.0f);
          if(weighting_strategy > 1) {
            ptheta[p] = new double[traj->size()];
            computePThetaBatch(*traj, ptheta[p], all_next_actions[p]);
          }
        }

        delete all_next_actions[p];
      
      
        //adjust q_targets
        i=0;
        for (auto it : *traj) {
          if(it.goal_reached)
            q_targets[p]->at(i) = it.r;
          else {
//             LOG_DEBUG(i << " " << traj->size() << " " << mini_batch_size);
            q_targets[p]->at(i) = it.r + gamma * q_targets[p]->at(i);
          }
          
          if(weighting_strategy==1)
            q_targets_weights[p]->at(i)=1.0f/it.p0;
          else if(weighting_strategy==2)
            q_targets_weights[p]->at(i)=ptheta[p][i]/it.p0;
          else if(weighting_strategy==3)
            q_targets_weights[p]->at(i)=std::min((double)1.0f, ptheta[p][i]/it.p0);
          
          i++;
        }
      }
      
      if(reset_qnn) {
        delete _qnn;
        _qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q,
                       alpha_v,
                       mini_batch_size,
                       decay_v,
                       hidden_layer_type, batch_norm,
                       weighting_strategy > 0);
      }
      
      std::vector<double> q_targets_final(traj->size());
      std::vector<double> q_targets_weights_final(traj->size());
      for(uint i=0;i<traj->size();i++){
        uint imax = 0;
        double bestS = q_targets[0]->at(i);
        for(uint p=1;p<_ann.size();p++)
          if(q_targets[p]->at(i) > bestS){
            bestS = q_targets[p]->at(i);
            imax = p;
          }
        q_targets_final[i]= bestS;
        if(weighting_strategy != 0)
          q_targets_weights_final[i] = q_targets_weights[imax]->at(i);
      }
      
      //Update critic
      if(weighting_strategy != 0)
        _qnn->stepCritic(all_states, all_actions, q_targets_final, iter, &q_targets_weights_final);
      else
        _qnn->stepCritic(all_states, all_actions, q_targets_final, iter);
      
      for(uint p=0;p < _ann.size(); p++){
        delete q_targets[p];
        if(weighting_strategy != 0) {
          delete q_targets_weights[p];
          if(weighting_strategy > 1)
            delete[] ptheta[p];
        }
      }
      
      //usefull?
      _qnn->ZeroGradParameters();
    }
  }

  void update_actor_critic() {
    if(!learning || trajectory.size() < mini_batch_size)
      return;

    if(trajectory.size() != mini_batch_size) {
      mini_batch_size = trajectory.size();
      for(uint mm=0; mm < multi_policies; mm++) {
        qnn[mm]->increase_batchsize(mini_batch_size);
        ann[mm]->increase_batchsize(mini_batch_size);
      }
    }

    std::vector<std::deque<sample>> _trajectories(multi_policies);
    if(force_same_data_mp && no_forgot_offline &&
        trajectory_noforgot.size() > trajectory.size() && trajectory_noforgot.size() > replay_memory) {
      for(uint mm=0; mm < multi_policies; mm++) {
        _trajectories[mm].resize(replay_memory);
        sample_transition(_trajectories[mm], trajectory_noforgot, replay_memory, ann[mm]);
      }
    }

    for(uint mm=0; mm < multi_policies; mm++) {
      if(!force_same_data_mp && no_forgot_offline && trajectory_noforgot.size() > trajectory.size()
          && trajectory_noforgot.size() > replay_memory) {
        trajectory.resize(replay_memory);
        sample_transition(trajectory, trajectory_noforgot, replay_memory, ann[mm]);
      } else if(force_same_data_mp && no_forgot_offline &&
                trajectory_noforgot.size() > trajectory.size() && trajectory_noforgot.size() > replay_memory) {
        trajectory = _trajectories[mm];
      }

      for(uint n=0; n<nb_fitted_updates; n++) {
        for(uint i=0; i<nb_critic_updates ; i++)
          critic_update(nb_internal_critic_updates, qnn[mm], ann[mm]);

        if(reset_ann) {
          delete ann[mm];
          ann[mm] = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, mini_batch_size, hidden_layer_type, last_layer_actor,
                            batch_norm);
        }

        for(uint i=0; i<nb_actor_updates ; i++)
          actor_update_grad(qnn[mm], ann[mm]);

//         if(!force_same_data_mp && no_forgot_offline && trajectory_noforgot.size() > trajectory.size()
//             && trajectory_noforgot.size() > replay_memory && mixed_sampling) {
//           trajectory.resize(replay_memory);
//           sample_transition(trajectory, trajectory_noforgot, replay_memory, ann[mm]);
//         }
      }
    }
    
    if(policy_selection == 2 || policy_selection == 12 || policy_selection == 4){
      for(uint i=0; i < std::max(nb_critic_updates, nb_fitted_updates) ; i++){
        if(no_forgot_offline && trajectory_noforgot.size() > trajectory.size()
          && trajectory_noforgot.size() > replay_memory){
          trajectory.resize(replay_memory);
          sample_transition(trajectory, trajectory_noforgot, replay_memory, nullptr);
        }
        global_qnn->increase_batchsize(trajectory.size());
        critic_global_update(nb_internal_critic_updates, global_qnn, ann);
      }
    }
  }

  void end_episode(bool) override {
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
  double criticEval(const std::vector<double>&, const std::vector<double>&) override {
    return 0;
  }

  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return nullptr;
  }

  void save(const std::string& path, bool, bool) override {
    ann[0]->save(path+".actor");
    qnn[0]->save(path+".critic");
    //      bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann[0]->load(path+".actor");
    qnn[0]->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward
#ifndef NDEBUG
        << " " << std::setw(8) << std::fixed << std::setprecision(5) << noise
        << " " << trajectory.size()
        << " " << ann[0]->weight_l1_norm()
        << " " << std::fixed << std::setprecision(7) << qnn[0]->error()
        << " " << qnn[0]->weight_l1_norm()
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

  double noise;
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint mini_batch_size;
  uint replay_memory, nb_actor_updates, nb_critic_updates, nb_fitted_updates, nb_internal_critic_updates;
  double alpha_a;
  double alpha_v;
  double decay_v;

  uint batch_norm, sampling_strategy, weighting_strategy, last_layer_actor,
       hidden_layer_type, multi_policies, policy_selection, policy_selected;
  bool learning, reset_qnn, inverting_grad;
  bool reset_ann, no_forgot_offline, force_same_data_mp;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::deque<sample> trajectory_noforgot;

  std::vector<MLP*> ann;
  std::vector<MLP*> qnn;
  std::vector<double> sum_onlineQSA;
  MLP* global_qnn;

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

