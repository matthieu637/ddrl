#ifndef OFFLINECACLAAG_HPP
#define OFFLINECACLAAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "nn/MLP.hpp"
#include "nn/DODevMLP.hpp"

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

  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }

    for (uint i = 0; i < a.size(); i++) {
      if(a[i] != b.a[i])
        return a[i] < b.a[i];
    }

    return false;
  }

} sample;

template<typename NN = MLP>
class NFACVMemAg : public arch::AACAgent<NN, arch::AgentProgOptions> {
 public:
  typedef NN PolicyImpl;

  NFACVMemAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<NN, arch::AgentProgOptions>(_nb_motors, _nb_sensors), nb_sensors(_nb_sensors), normst(_nb_sensors,
        0.f), empty_action(0) {

  }

  virtual ~NFACVMemAg() {
    delete vnn;
    delete ann;

    delete qnn;
    delete qnn_target;
    delete ann_best;
    if(newidea > 0)
      delete ann_behav;
    if(smooth_udpate_mem)
      delete ann_smooth;

    delete ann_testing;
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action = ann_testing->computeOut(sensors);
    
    if(learning && newidea == 2 && last_action.get() != nullptr){
      //might be more adapted to sparse reward?
      auto ac_explo = ann_behav->computeOut(last_state);
      reward = reward + beta * (1.f - l2dista(*last_action,*ac_explo));
      delete ac_explo;
    }

    if (last_action.get() != nullptr && learning) {
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached};
      insertSample(sa);
      trajectory.push_back(sa);
    }

    last_pure_action.reset(new vector<double>(*next_action));
//     RMME
//     double qq=bib::Utils::rand01();
//     if(qq < 0.33)
//       next_action->at(0) = -1;
//     else if(qq > 0.66)
//       next_action->at(0) = 1;
//     else
//       next_action->at(0) = (2*bib::Utils::rand01() - 1.f)*0.85;
//
//     if(episode == 0) {
//       LOG_FILE_NNL("pi.data", "");
//       for(double m : *next_action)
//         LOG_FILE_NNL("pi.data", m << " ");
//       LOG_FILE("pi.data", "");
//     }
//     if(false) {
    if(learning) {
      if(newidea == 0) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;

        if(exploration_strat > 0 && (ann_best != nullptr || qoffofcurrentpol )) {
          double qbestpol_eval_exploration = qnn_target->computeOutVF(sensors, *next_action);
          if(exploration_strat == 1) {
            auto ac_test = ann_best->computeOut(sensors);
            double qbestpol_eval_exploitation = qnn_target->computeOutVF(sensors, *ac_test);
            if(qbestpol_eval_exploration > qbestpol_eval_exploitation)
              delete ac_test;
            else {
              delete next_action;
              next_action = ac_test;
            }
          } else if(exploration_strat == 2) {
            qnn->computeOutVF(sensors, *next_action);
            const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
            double* q_values_diff = q_values_blob->mutable_cpu_diff();
            q_values_diff[q_values_blob->offset(0,0,0,0)] = -1.0f;
            qnn->critic_backward();
            const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
            auto ac_diff_critic = critic_action_blob->cpu_diff();
            for (uint i = 0; i < this->nb_motors ; i++) {
              next_action->at(i) += 0.5 * ac_diff_critic[i];
              if(next_action->at(i) > 1.0)
                next_action->at(i) = 1.;
              else if(next_action->at(i) < -1.0)
                next_action->at(i) = -1.;
            }
          } else if(exploration_strat == 3) {
            qnn->computeOutVF(sensors, *next_action);
            const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
            double* q_values_diff = q_values_blob->mutable_cpu_diff();
            q_values_diff[q_values_blob->offset(0,0,0,0)] = -1.0f;
            qnn->critic_backward();
            const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
            auto ac_diff_critic = critic_action_blob->cpu_diff();
            for (uint i = 0; i < this->nb_motors ; i++) {
              next_action->at(i) -= 0.5 * ac_diff_critic[i];
              if(next_action->at(i) > 1.0)
                next_action->at(i) = 1.;
              else if(next_action->at(i) < -1.0)
                next_action->at(i) = -1.;
            }
          }
        }
      } else if(newidea == 1 && bib::Utils::rand01() < noise){
        delete next_action;
        next_action = ann_behav->computeOut(sensors);
      } else if(newidea == 2){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(newidea == 3){
        auto ac_explo = ann_behav->computeOut(sensors);
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*ac_explo, noise);
        delete next_action;
        delete ac_explo;
        next_action = randomized_action;
      } else if(newidea == 4){
        auto ac_explo = ann_behav->computeOut(sensors);
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*ac_explo, noise);
        for(uint j=0;j<this->nb_motors;j++){
          if(next_action->at(j) <= randomized_action->at(j) && randomized_action->at(j) <= ac_explo->at(j))
            next_action->at(j) = randomized_action->at(j);
          else if(next_action->at(j) <= randomized_action->at(j) && randomized_action->at(j) >= ac_explo->at(j) && next_action->at(j) <= ac_explo->at(j) )
            next_action->at(j) = ac_explo->at(j);
          else if(next_action->at(j) <= randomized_action->at(j) && randomized_action->at(j) >= ac_explo->at(j) && next_action->at(j) >= ac_explo->at(j) 
            && ac_explo->at(j) <= next_action->at(j) - (randomized_action->at(j) - next_action->at(j)))
            next_action->at(j) = next_action->at(j) - (randomized_action->at(j) - next_action->at(j));
          else if(next_action->at(j) <= randomized_action->at(j) && randomized_action->at(j) >= ac_explo->at(j) && next_action->at(j) >= ac_explo->at(j) 
            && ac_explo->at(j) >= next_action->at(j) - (randomized_action->at(j) - next_action->at(j)))
            next_action->at(j) = ac_explo->at(j);
          else if(next_action->at(j) >= randomized_action->at(j) && randomized_action->at(j) >= ac_explo->at(j))
            next_action->at(j) = randomized_action->at(j);
          else if(next_action->at(j) >= randomized_action->at(j) && randomized_action->at(j) <= ac_explo->at(j) && next_action->at(j) >= ac_explo->at(j))
            next_action->at(j) = ac_explo->at(j);
          else if(next_action->at(j) >= randomized_action->at(j) && randomized_action->at(j) <= ac_explo->at(j) && next_action->at(j) <= ac_explo->at(j)
            && ac_explo->at(j) >= next_action->at(j) + (next_action->at(j) - randomized_action->at(j)) )
            next_action->at(j) = next_action->at(j) + (next_action->at(j) - randomized_action->at(j));
          else if(next_action->at(j) >= randomized_action->at(j) && randomized_action->at(j) <= ac_explo->at(j) && next_action->at(j) <= ac_explo->at(j)
            && ac_explo->at(j) <= next_action->at(j) + (next_action->at(j) - randomized_action->at(j)) )
            next_action->at(j) = ac_explo->at(j);
            
          if(next_action->at(j) > 1.0)
            next_action->at(j) = 1.;
          else if(next_action->at(j) < -1.0)
            next_action->at(j) = -1.;
        }
          
        delete randomized_action;
        delete ac_explo;
      }
    }
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    update_delta_neg        = pt->get<bool>("agent.update_delta_neg");
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
    kMinibatchSize          = pt->get<uint>("agent.mini_batch_size");
    tau_soft_update         = pt->get<double>("agent.tau_soft_update");
    replay_memory           = pt->get<uint>("agent.replay_memory");
    smooth_udpate_mem       = pt->get<bool>("agent.smooth_udpate_mem");
    qoffofcurrentpol        = pt->get<bool>("agent.qoffofcurrentpol");
    newidea                 = pt->get<uint>("agent.newidea");
    spacedist               = pt->get<uint>("agent.spacedist");
    exploration_strat       = pt->get<uint>("agent.exploration_strat");
    beta                    = pt->get<double>("agent.beta");
    corrected_update_ac     = false;
    gae                     = false;
    inverting_gradient      = false;
    best_learning_perf      = std::numeric_limits<double>::lowest();
    try {
      corrected_update_ac   = pt->get<bool>("agent.corrected_update_ac");
    } catch(boost::exception const& ) {
    }
    try {
      inverting_gradient   = pt->get<bool>("agent.inverting_gradient");
    } catch(boost::exception const& ) {
    }
    if(corrected_update_ac) {
      try {
        corrected_update_ac_factor   = pt->get<double>("agent.corrected_update_ac_factor");
      } catch(boost::exception const& ) {
      }
    }

    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");

    if(lambda >=0. && batch_norm_critic != 0) {
      LOG_DEBUG("to be done!");
      exit(1);
    }

    ann = new NN(nb_sensors, *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type,
                 batch_norm_actor, true);
    if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);

    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic);
    if(std::is_same<NN, DODevMLP>::value)
      vnn->exploit(pt, ann);

    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    if(batch_norm_critic != 0)
      vnn_testing = new NN(*vnn, false, ::caffe::Phase::TEST);

    qnn = new NN(nb_sensors + this->nb_motors, nb_sensors, *hidden_unit_v, alpha_v, kMinibatchSize,
                 -1, hidden_layer_type, batch_norm_critic, false);

    qnn_target = new NN(*qnn, false);

    if(std::is_same<NN, DODevMLP>::value) {
      try {
        if(pt->get<bool>("devnn.reset_learning_algo")) {
          LOG_ERROR("NFAC cannot reset anything with DODevMLP");
          exit(1);
        }
      } catch(boost::exception const& ) {
      }
    }

    ann_best = nullptr;
    if(smooth_udpate_mem) {
      ann_smooth = new NN(*ann, false);
      ann_smooth->increase_batchsize(kMinibatchSize);
    }
    if(newidea > 0) {
      ann_behav = new NN(*ann, true);
      ann_behav->increase_batchsize(kMinibatchSize);
    }
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    trajectory.clear();

//     if(std::is_same<NN, DODevMLP>::value) {
//       static_cast<DODevMLP *>(vnn)->inform(episode, this->last_sum_weighted_reward);
//       static_cast<DODevMLP *>(ann)->inform(episode, this->last_sum_weighted_reward);
//       static_cast<DODevMLP *>(ann_testing)->inform(episode, this->last_sum_weighted_reward);
//     }

    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
  }

  void update_critic() {
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
        if(batch_norm_critic != 0) {
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
        if(vnn_from_scratch) {
          delete vnn;
          vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, trajectory.size(), -1, hidden_layer_type,
                       batch_norm_critic);
        }
        if(lambda < 0.f && batch_norm_critic == 0)
          vnn->learn_batch(all_states, empty_action, v_target, stoch_iter_critic);
        else if(lambda < 0.f) {
          for(uint sia = 0; sia < stoch_iter_critic; sia++) {
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
            for (auto it : trajectory) {
              q_values_diff[i] = (all_V->at(i)-v_target[i])/s;
              i++;
            }
            vnn->critic_backward();
            vnn->getSolver()->ApplyUpdate();
            vnn->getSolver()->set_iter(vnn->getSolver()->iter() + 1);
            delete all_V;
          }
        } else {
          auto all_V = vnn->computeOutVFBatch(all_states, empty_action);
          std::vector<double> deltas(trajectory.size());
//
//        Simple computation for lambda return
//
          li=0;
          for (auto it : trajectory) {
            deltas[li] = v_target[li] - all_V->at(li);
            ++li;
          }

          std::vector<double> diff(trajectory.size());
          li=0;
          for (auto it : trajectory) {
            diff[li] = 0;
            for (uint n=li; n<trajectory.size(); n++)
              diff[li] += std::pow(this->gamma * lambda, n-li) * deltas[n];
            li++;
          }
          ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");

// //           comment following lines to compare with the other formula
          li=0;
          for (auto it : trajectory) {
            diff[li] = diff[li] + all_V->at(li);
            ++li;
          }

          vnn->learn_batch(all_states, empty_action, diff, stoch_iter_critic);
// //
// //        The mechanic formula
// //
//           std::vector<double> diff2(trajectory.size());
//           li=0;
//           for (auto it : trajectory){
//             diff2[li] = 0;
//             double sum_n = 0.f;
//             for(int n=1;n<=((int)trajectory.size()) - li - 1;n++){
//               double sum_i = 0.f;
//               for(int i=li;i<=li+n-1;i++)
//                 sum_i += std::pow(this->gamma, i-li) * trajectory[i].r;
//               sum_i += std::pow(this->gamma, n) * all_nextV->at(li+n-1);
//               sum_i *= pow(lambda, n-1);
//               sum_n += sum_i;
//             }
//             sum_n *= (1.f-lambda);
//
//             double sum_L = 0.f;
//             for(int i=li;i<(int)trajectory.size();i++)
//               sum_L += std::pow(this->gamma, i-li) * trajectory[i].r;
//             if(trajectory[trajectory.size()-1].goal_reached)
//               sum_n += std::pow(lambda, trajectory.size() - li - 1) * sum_L;
//             else {
//               sum_L += std::pow(this->gamma, ((int)trajectory.size()) - li) * all_nextV->at(trajectory.size()-1);
//               sum_n += std::pow(lambda, trajectory.size() - li - 1) * sum_L;
//             }
//
//             sum_n -= all_V->at(li);
//
//             diff2[li] = sum_n;
//             ++li;
//           }
//           bib::Logger::PRINT_ELEMENTS(diff, "form1 ");
//           bib::Logger::PRINT_ELEMENTS(diff2, "mech form ");
//
//           if(trajectory[trajectory.size()-1].goal_reached)
//             exit(1);
          delete all_V;
        }

        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();
    }
  }

  void sample_transition(std::vector<sample>& traj, const std::deque<sample>& from) {
    for(uint i=0; i<traj.size(); i++) {
      int r = std::uniform_int_distribution<int>(0, from.size() - 1)(*bib::Seed::random_engine());
      traj[i] = from[r];
    }
  }

  void insertSample(const sample& sa) {
    if(all_sample.size() >= replay_memory)
      all_sample.pop_front();
    all_sample.push_back(sa);

    if(newidea > 0) {
      behaviorpolicy_update();
      return;
    }

    if(ann_best == nullptr && !qoffofcurrentpol)
      return;

    if(qoffofcurrentpol) {
      ann->increase_batchsize(kMinibatchSize);
      if(!smooth_udpate_mem)
        online_update_qoff(ann);
      else if(smooth_udpate_mem)
        online_update_qoff(ann_smooth);

      return;
    }

    if(!smooth_udpate_mem)
      online_update_qoff(ann_best);
    else if(smooth_udpate_mem)
      online_update_qoff(ann_smooth);
  }

  void online_update_qoff(NN *ann_target) {

    if(all_sample.size() < kMinibatchSize)
      return;

    std::vector<sample> traj(kMinibatchSize);
    sample_transition(traj, all_sample);

    //compute \pi(s_{t+1})
    std::vector<double> all_next_states(traj.size() * nb_sensors);
    std::vector<double> all_states(traj.size() * nb_sensors);
    std::vector<double> all_actions(traj.size() * this->nb_motors);
    uint i=0;
    for (auto it : traj) {
      std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
      std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
      std::copy(it.a.begin(), it.a.end(), all_actions.begin() + i * this->nb_motors);
      i++;
    }

    auto all_next_actions = ann_target->computeOutBatch(all_next_states);

    //compute next q
    auto q_targets = qnn_target->computeOutVFBatch(all_next_states, *all_next_actions);
    delete all_next_actions;

    //adjust q_targets
    i=0;
    for (auto it : traj) {
      if(it.goal_reached)
        q_targets->at(i) = it.r;
      else
        q_targets->at(i) = it.r + this->gamma * q_targets->at(i);

      i++;
    }

    //Update critic
    qnn->learn_batch(all_states, all_actions, *q_targets, 1);

    // Soft update of targets networks
    qnn_target->soft_update(*qnn, tau_soft_update);
    if(smooth_udpate_mem) {
      if(!qoffofcurrentpol)
        ann_smooth->soft_update(*ann_best, tau_soft_update);
      else
        ann_smooth->soft_update(*ann, tau_soft_update);

    }

    delete q_targets;
  }

  void behaviorpolicy_update() {

    if(all_sample.size() < kMinibatchSize)
      return;

    std::vector<sample> traj(kMinibatchSize);
    sample_transition(traj, all_sample);

    //compute \pi(s_{t+1})
    std::vector<double> all_states(traj.size() * nb_sensors);
    int i=0;
    for (auto it : traj) {
      std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
      i++;
    }

    std::vector<double> weights(traj.size()*traj.size(), 1.f);
    if(spacedist == 0){
      i=0;
      for (auto s0 : traj) {
        for (auto st : traj) {
          l2distupdate(s0.s, st.s);
          i++;
        }
      }
      i=0;
      for (auto s0 : traj) {
        for (auto st : traj) {
          weights[i] = 1.f - l2dist(s0.s, st.s);
          if(weights[i]<= 0.000001)
            weights[i]=0.000001;
          i++;
        }
      }
    } else if(spacedist == 1){
      i=0;
      for (auto s0 : traj) {
        for (auto st : traj) {
          weights[i] = 1.f - l2dist(s0.a, st.a);
          if(weights[i]<= 0.000001)
            weights[i]=0.000001;
          i++;
        }
      }
    }

    auto all_actions_out = ann_behav->computeOutBatch(all_states);
// #warning RMEM
//     if(episode > 2) {
//       LOG_FILE_NNL("be.data", "");
//       i=0;
//       for(double m : *all_actions_out) {
//         LOG_FILE_NNL("be.data", m << " ");
//         if(i % this->nb_motors == this->nb_motors - 1 && i != 0)
//           LOG_FILE("be.data", "");
//         i++;
//       }
//       exit(1);
//     }
    const auto actor_actions_blob = ann_behav->getNN()->blob_by_name(MLP::actions_blob_name);
    auto ac_diff = actor_actions_blob->mutable_cpu_diff();
    uint k=0;
    for (i=0; i<actor_actions_blob->count(); i++) {
      std::vector<double> score;
      std::vector<double> all_diff;
      uint motors_dimension = i % this->nb_motors;
      uint j=k;
      for (auto st : traj) {
        double x = st.a[motors_dimension] - all_actions_out->at(i);
        score.push_back(-(x*x)/weights[j]);
        all_diff.push_back(-x/weights[j]);
        j++;
      }
      if(motors_dimension == this->nb_motors - 1 ) {
        k += traj.size();
      }

      uint winner = std::distance(score.begin(), std::max_element(score.begin(), score.end()));
      ac_diff[i] = -all_diff[winner];
    }
    ann_behav->actor_backward();
    ann_behav->getSolver()->ApplyUpdate();
    ann_behav->getSolver()->set_iter(ann_behav->getSolver()->iter() + 1);
    delete all_actions_out;
  }

  void l2distupdate(const std::vector<double>& a, const std::vector<double>& b) {
    for(uint i=0; i<a.size(); i++) {
      double d = (a[i] - b[i]);
      if(d*d > normst[i])
        normst[i]= d*d;
    }
  }

  double l2dist(const std::vector<double>& a, const std::vector<double>& b) const {
    double r = 0.f;
    for(uint i=0; i<a.size(); i++) {
      double d = (a[i] - b[i]);
      r += (d*d)/normst[i];
    }
    return r/((double) a.size());
  }
  
  double l2dista(const std::vector<double>& a, const std::vector<double>& b) const {
    double r = 0.f;
    for(uint i=0; i<a.size(); i++) {
      double d = (a[i] - b[i]);
      r += (d*d);
    }
    return sqrt(r)/(2.f * (double) a.size());
  }

  void end_episode(bool learning) override {
//     LOG_FILE("policy_exploration", ann->hash());
    if(!learning)
      return;
// #warning RMME
//     return;

    if(this->sum_weighted_reward > best_learning_perf) {
      if (ann_best != nullptr)
        delete ann_best;
      ann_best = new NN(*ann, false);
      ann_best->increase_batchsize(kMinibatchSize);
      best_learning_perf = this->sum_weighted_reward;
    }

    if(trajectory.size() > 0) {
      vnn->increase_batchsize(trajectory.size());
      if(batch_norm_critic != 0)
        vnn_testing->increase_batchsize(trajectory.size());
    }

    if(update_critic_first)
      update_critic();

    if (trajectory.size() > 0) {
      std::vector<double> sensors(trajectory.size() * nb_sensors);
      std::vector<double> actions(trajectory.size() * this->nb_motors);
      std::vector<bool> disable_back(trajectory.size() * this->nb_motors, false);
      const std::vector<bool> disable_back_ac(this->nb_motors, true);
      std::vector<double> deltas_blob(trajectory.size() * this->nb_motors);
      std::vector<double> deltas(trajectory.size());

      std::vector<double> all_states(trajectory.size() * nb_sensors);
      std::vector<double> all_next_states(trajectory.size() * nb_sensors);
      uint li=0;
      for (auto it : trajectory) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
        li++;
      }

      decltype(vnn->computeOutVFBatch(all_next_states, empty_action)) all_nextV, all_mine;
      if(batch_norm_critic != 0) {
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
      for (auto it : trajectory) {
        sample sm = it;
        double v_target = sm.r;
        if (!sm.goal_reached) {
          double nextV = all_nextV->at(li);
          v_target += this->gamma * nextV;
        }

        deltas[li] = v_target - all_mine->at(li);
        ++li;
      }

      if(gae) {
        //
        //        Simple computation for lambda return
        //
        std::vector<double> diff(trajectory.size());
        li=0;
        for (auto it : trajectory) {
          diff[li] = 0;
          for (uint n=li; n<trajectory.size(); n++)
            diff[li] += std::pow(this->gamma * lambda, n-li) * deltas[n];
          li++;
        }

        ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");
        li=0;
        for (auto it : trajectory) {
//           diff[li] = diff[li] + all_V->at(li);
          deltas[li] = diff[li];
          ++li;
        }
      }

      uint n=0;
      li=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        if(deltas[li] > 0.) {
          std::copy(it->a.begin(), it->a.end(), actions.begin() + li * this->nb_motors);
          n++;
        } else if(update_delta_neg) {
          std::copy(it->pure_a.begin(), it->pure_a.end(), actions.begin() + li * this->nb_motors);
        } else {
          std::copy(it->a.begin(), it->a.end(), actions.begin() + li * this->nb_motors);
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
        li++;
      }

      if(n > 0) {
        for(uint sia = 0; sia < stoch_iter_actor; sia++) {
          ann->increase_batchsize(trajectory.size());
          //learn BN
          auto ac_out = ann->computeOutBatch(sensors);
          if(batch_norm_actor != 0) {
            //re-compute ac_out with BN as testing
            double* weights = new double[ann->number_of_parameters(false)];
            ann->copyWeightsTo(weights, false);
            ann_testing->copyWeightsFrom(weights, false);
            delete[] weights;
            delete ac_out;
            ann_testing->increase_batchsize(trajectory.size());
            ac_out = ann_testing->computeOutBatch(sensors);
          }

          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          auto ac_diff = actor_actions_blob->mutable_cpu_diff();
          for(int i=0; i<actor_actions_blob->count(); i++) {
            if(disable_back[i]) {
              ac_diff[i] = 0.00000000f;
            } else {
              double x = actions[i] - ac_out->at(i);
              if(!corrected_update_ac) {
                ac_diff[i] = -x;
                if(inverting_gradient) {
                  const double min_ = -1.0;
                  const double max_ = 1.0;

                  if (ac_diff[i] < 0)
                    ac_diff[i] *= (max_ - ac_out->at(i)) / (max_ - min_);
                  else if (ac_diff[i] > 0)
                    ac_diff[i] *= (ac_out->at(i) - min_) / (max_ - min_);
                }
              } else {
                double fabs_x = fabs(x);
                if(fabs_x <= corrected_update_ac_factor)
                  ac_diff[i] = sign(x) * sign(deltas_blob[i]) * (sqrt(fabs_x)
                               - sqrt(corrected_update_ac_factor) - sign(deltas_blob[i]) * deltas_blob[i]/corrected_update_ac_factor );
                else
                  ac_diff[i] = -deltas_blob[i] / x;
              }
            }
          }
          ann->actor_backward();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
          delete ac_out;
        }
      } else if(batch_norm_actor != 0) {
        ann->increase_batchsize(trajectory.size());
        delete ann->computeOutBatch(sensors);
      }

      delete all_nextV;
      delete all_mine;

      if(batch_norm_actor != 0)
        ann_testing->increase_batchsize(1);
    }

    if(!update_critic_first) {
      update_critic();
    }
  }

  void end_instance(bool learning) override {
    if(learning)
      episode++;
  }

  void save(const std::string& path, bool, bool) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
  }

  void save_run() override {
    ann->save("continue.actor");
    vnn->save("continue.critic");
    struct algo_state st = {episode};
    bib::XMLEngine::save(st, "algo_state", "continue.algo_state.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

  void load_previous_run() override {
    ann->load("continue.actor");
    vnn->load("continue.critic");
    auto p3 = bib::XMLEngine::load<struct algo_state>("algo_state", "continue.algo_state.data");
    episode = p3->episode;
    delete p3;
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size();
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
        this->sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }

  double sign(double x) {
    if(x>=0)
      return 1.f;
    return -1.f;
  }

 private:
  uint nb_sensors;
  uint episode = 0;

  double noise;
  bool vnn_from_scratch, update_critic_first,
       update_delta_neg, corrected_update_ac, gae;
  bool inverting_gradient;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type;
  double lambda, corrected_update_ac_factor;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<sample> trajectory;
  std::deque<sample> all_sample;

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* vnn_testing;

//   q part
  uint kMinibatchSize, replay_memory;
  double tau_soft_update;
  NN* qnn, *qnn_target;

//   mem part
  NN* ann_best, *ann_smooth, *ann_behav;
  double best_learning_perf;
  bool smooth_udpate_mem, qoffofcurrentpol;
  uint exploration_strat, newidea, spacedist;
  std::vector<double> normst;
  double beta;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference

  struct algo_state {
    uint episode;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) {
      ar& BOOST_SERIALIZATION_NVP(episode);
    }
  };
};

#endif

