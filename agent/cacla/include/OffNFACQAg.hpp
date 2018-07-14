#ifndef OFFNFACQAG_HPP
#define OFFNFACQAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include "arch/ARLAgent.hpp"
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
  double dpmu;

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

typedef struct _trajectory {
  std::shared_ptr<std::deque<sample>> transitions;
  double rewards;
} trajectory;

template<typename NN = MLP>
class OffNFACQAg : public arch::ARLAgent<arch::AgentProgOptions> {
 public:
  typedef NN PolicyImpl;

  OffNFACQAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::ARLAgent<arch::AgentProgOptions>(_nb_motors, _nb_sensors), empty_action(0) {

  }

  virtual ~OffNFACQAg() {
    delete qnn;
    delete ann;

    delete ann_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action = ann_testing->computeOut(sensors);

    if (last_action.get() != nullptr && learning) {
      double p0 = 1.f;
      for(uint i=0; i < this->nb_motors; i++) {
        p0 *= bib::Proba<double>::truncatedGaussianDensity(last_action->at(i), last_pure_action->at(i), noise);
      }
      trajectories.back()->transitions->push_back( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0});
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


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
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
    max_trajectory          = pt->get<uint>("agent.max_trajectory");
    offpolicy_strategy      = pt->get<uint>("agent.offpolicy_strategy");
    add_v_corrector         = pt->get<bool>("agent.add_v_corrector");
    offpolicy_actor         = pt->get<bool>("agent.offpolicy_actor");
    number_global_fitted_iteration         = pt->get<uint>("agent.number_global_fitted_iteration");
    offpolicy_critic                       = pt->get<bool>("agent.offpolicy_critic");
    shuffle_buffer                         = pt->get<bool>("agent.shuffle_buffer");
    determinist_update                     = pt->get<bool>("agent.determinist_update");

    gamma_corrector                        = pt->get<bool>("agent.gamma_corrector");
    batch_norm_testing                     = pt->get<bool>("agent.batch_norm_testing");
    use_qmu                                = pt->get<int>("agent.use_qmu");
    use_qmu_actor                          = pt->get<bool>("agent.use_qmu_actor");
    ac_onpolnexta                        = pt->get<bool>("agent.ac_onpolnexta");
    uint momentum           = pt->get<uint>("agent.momentum");
    reset_actor                          = pt->get<bool>("agent.reset_actor");
    
    gae                     = false;

    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");

    if(offpolicy_actor && gae && stoch_iter_actor > 1) {
      LOG_DEBUG("to be done! (stoch_iter_actor>1 with gae and offpol)");
      exit(1);
    }
    
    if(use_qmu_actor && determinist_update) {
      LOG_DEBUG("use_qmu_actor is useless with determinist update");
      exit(1);
    }
    
    if(use_qmu_actor && use_qmu<=0) {
      LOG_DEBUG("use_qmu_actor with use_qmu<=0");
      exit(1);
    }
    
    if(ac_onpolnexta && use_qmu >= -1) {
      LOG_DEBUG("ac_onpolnexta with use_qmu >= -1");
      exit(1);
    }
    
    if(offpolicy_critic && use_qmu < -1)
      LOG_INFO("WARNING : you learn onpolicy Q^\\mu with off_policy correction");
    
    if(!offpolicy_critic && use_qmu == -1)
      LOG_INFO("WARNING : you learn offpolicy Q^\\pi without off_policy correction");

    ann = new NN(this->get_state_size(), *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type,
                 actor_output_layer_type,
                 batch_norm_actor, true, momentum);
    if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);

    qnn = new NN(this->get_state_size() + this->nb_motors, this->get_state_size(), *hidden_unit_v, alpha_v, 1, -1,
                 hidden_layer_type, batch_norm_critic,
                 add_v_corrector, momentum);
    if(std::is_same<NN, DODevMLP>::value)
      qnn->exploit(pt, ann);

    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);


    if(std::is_same<NN, DODevMLP>::value) {
      try {
        if(pt->get<bool>("devnn.reset_learning_algo")) {
          LOG_ERROR("NFAC cannot reset anything with DODevMLP");
          exit(1);
        }
      } catch(boost::exception const& ) {
      }
    }
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    if(learning) {
      if(trajectories.size() >= max_trajectory)
        trajectories.pop_front();

      trajectory* t = new trajectory;
      t->transitions.reset(new std::deque<sample>);
      t->rewards = 0;
      trajectories.push_back(std::shared_ptr<trajectory>(t));
    }

//     if(std::is_same<NN, DODevMLP>::value) {
//       static_cast<DODevMLP *>(qnn)->inform(episode, last_sum_weighted_reward);
//       static_cast<DODevMLP *>(ann)->inform(episode, last_sum_weighted_reward);
//       static_cast<DODevMLP *>(ann_testing)->inform(episode, last_sum_weighted_reward);
//     }

    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
  }

  void update_critic() {
    int all_size = alltransitions();
    if (all_size > 0) {
      //remove trace of old policy
      std::vector<double> all_states(all_size * this->get_state_size());
      std::vector<double> all_next_states(all_size * this->get_state_size());
      std::vector<double> all_actions(all_size * this->nb_motors);

      int li=0;
      for(auto one_trajectory : trajectories) {
        const std::deque<sample>& trajectory = *one_trajectory->transitions;

        for (auto it : trajectory) {
          std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * this->get_state_size());
          std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * this->get_state_size());
          std::copy(it.a.begin(), it.a.end(), all_actions.begin() + li * this->nb_motors);
          li++;
        }
      }
      ASSERT(li == all_size, "pb");

      ann->increase_batchsize(all_size);
      std::vector<double>* all_next_actions;
      if(use_qmu >= -1)
        all_next_actions = ann->computeOutBatch(all_next_states);
      else {//use_qmu -2
        all_next_actions = new std::vector<double>(all_size * this->nb_motors);
        li=0;
        for(auto one_trajectory : trajectories) {
          const std::deque<sample>& trajectory = *one_trajectory->transitions;
          
          bool skip_first=true;
          for (auto it : trajectory) {
            if(skip_first){
              skip_first=false;
              continue;
            }
            std::copy(it.a.begin(), it.a.end(), all_next_actions->begin() + li * this->nb_motors);
            li++;
          }
          
          vector<double>* next_action = ann_testing->computeOut(trajectory.back().s);
          std::copy(next_action->begin(), next_action->end(), all_next_actions->begin() + li * this->nb_motors);
          delete next_action;
          li++;
        }
        
        ASSERT(li == all_size, "pb");
      }
      std::vector<double>* all_pi;
      if(lambda >= 0.f && offpolicy_strategy != 0 && offpolicy_critic )
        all_pi = ann->computeOutBatch(all_states);

      double* ptheta;
      double max_ptheta;
      std::vector<double>* sample_weight;
      if(lambda >= 0.f) {
        if(offpolicy_strategy != 0 && offpolicy_critic) {
          ptheta = new double[all_size];

          uint i=0;
          if(offpolicy_strategy >= 1 && offpolicy_strategy <= 3) {
            for(auto one_trajectory : trajectories) {
              const std::deque<sample>& trajectory = *one_trajectory->transitions;
              for (auto it : trajectory) {
                double p0 = 1.f;
                for(uint j=0; j < this->nb_motors; j++)
                  p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_pi->at(i*this->nb_motors+j), noise);

                ptheta[i] = p0;
                i++;
              }
            }
            max_ptheta = *std::max_element(ptheta, ptheta+all_size);
          }

          if(add_v_corrector)
            sample_weight = new std::vector<double>(all_size);
        }
      }

      auto iter = [&]() {
        std::vector<double> v_target(all_size);
        std::vector<double>* all_nextV;
        if(use_qmu <= 0)
          all_nextV = qnn->computeOutVFBatch(all_next_states, *all_next_actions);
        else {
          std::vector<double> all_actions_mu(all_size * this->nb_motors * use_qmu);
          std::vector<double> all_states_tmp(all_size * this->get_state_size() * use_qmu);
          for(int j=0;j<use_qmu;j++){
            vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*all_next_actions, noise);
            std::copy(randomized_action->begin(), randomized_action->end(), all_actions_mu.begin() + j * all_size * this->nb_motors);
            std::copy(all_next_states.begin(), all_next_states.end(), all_states_tmp.begin() + j * all_size * this->get_state_size());
            delete randomized_action;
          }
          qnn->increase_batchsize(all_size * use_qmu);
          all_nextV = qnn->computeOutVFBatch(all_states_tmp, all_actions_mu);
          qnn->increase_batchsize(all_size);
          double factor_ = use_qmu;
          for(int j=0;j<all_size;j++){
            double sum_ = 0.f;
            for(int k=0;k<use_qmu;k++)
              sum_ += all_nextV->at(k*all_size+j);
            all_nextV->at(j) = sum_ / factor_;
          }
        }

        li=0;
        for(auto one_trajectory : trajectories) {
          const std::deque<sample>& trajectory = *one_trajectory->transitions;
          for (auto it : trajectory) {
            double target = it.r;
            if (!it.goal_reached) {
              double nextV = all_nextV->at(li);
              target += this->gamma * nextV;
            }

            v_target[li] = target;
            li++;
          }
        }
        ASSERT(li == all_size, "pb");
//         bib::Logger::PRINT_ELEMENTS(v_target, "v_target");

        if(vnn_from_scratch) {
          delete qnn;
          qnn = new NN(this->get_state_size() + this->nb_motors, this->get_state_size(), *hidden_unit_v, alpha_v, all_size, -1,
                       hidden_layer_type,
                       batch_norm_critic, add_v_corrector);
        }
        if(lambda < 0.f)
          qnn->learn_batch(all_states, all_actions, v_target, stoch_iter_critic);
        else {
          auto all_V = qnn->computeOutVFBatch(all_states, all_actions);
          std::vector<double> deltas(all_size);
          //
          //        Simple computation for lambda return
          //
          for (int i=0; i<all_size; i++)
            deltas[i] = v_target[i] - all_V->at(i);

          std::vector<double> diff(all_size);
          int index_shift=0;
          for(auto one_trajectory : trajectories) {
            const std::deque<sample>& trajectory = *one_trajectory->transitions;
            li=0;
            for (auto it : trajectory) {
              diff[index_shift+li] = 0;
//               offpolicy_strategy
//               0 : lambda only
//               1 : lambda \pi (TB)
//               2 : \pi / \mu (IS)
//               3 : lambda min(1,\pi / \mu) (Retrace)
//               4 : lambda * (1-||a_t - \pi||) (ours)
//               5 : lambda * (1-min(||a_t - \pi||, ||u_t - \pi|| )) (ours)
              if(offpolicy_strategy == 0 || !offpolicy_critic) {
                for (uint n=li; n<trajectory.size(); n++)
                  diff[index_shift+li] += std::pow(this->gamma * lambda, n-li) * deltas[index_shift+n];
              } else if(offpolicy_strategy == 1) {
//                 for (int n=trajectory.size()-1; n>=li; n--){
//                   double ci = 1.f;
//                   for(int subn = 0;subn<n-li;subn++)
//                     ci = ci * lambda;
//                   diff[index_shift+li] += std::pow(this->gamma, n-li) * ci * deltas[index_shift+n];
//                 }
                double target_sum = 0;
                for (int n=trajectory.size()-1; n>=li; n--) {
                  target_sum += deltas[index_shift+n];
                  diff[index_shift+li] = target_sum;
                  target_sum *= this->gamma * lambda * (ptheta[index_shift+n]/max_ptheta);
                }
                if(add_v_corrector)
                  sample_weight->at(index_shift+li) = ptheta[index_shift+li]/max_ptheta;
              } else if(offpolicy_strategy == 2) {
                double target_sum = 0;
                for (int n=trajectory.size()-1; n>=li; n--) {
                  target_sum += deltas[index_shift+n];
                  diff[index_shift+li] = target_sum;
                  target_sum *= this->gamma * lambda * (ptheta[index_shift+n]/trajectory[n].dpmu);
                }
                if(add_v_corrector)
                  sample_weight->at(index_shift+li) = ptheta[index_shift+li]/trajectory[li].dpmu;
              } else if(offpolicy_strategy == 3) {
                double target_sum = 0;
                for (int n=trajectory.size()-1; n>=li; n--) {
                  target_sum += deltas[index_shift+n];
                  diff[index_shift+li] = target_sum;
                  target_sum *= this->gamma * lambda * std::min((double)1.f,ptheta[index_shift+n]/trajectory[n].dpmu);
                }
                if(add_v_corrector)
                  sample_weight->at(index_shift+li) = std::min((double)1.f,ptheta[index_shift+li]/trajectory[li].dpmu);
              } else if(offpolicy_strategy == 4) {
                double target_sum = 0;
                for (int n=trajectory.size()-1; n>=li; n--) {
                  target_sum += deltas[index_shift+n];
                  diff[index_shift+li] = target_sum;
                  target_sum *= this->gamma * lambda * (1.f - l2dist(trajectory[n].a, *all_pi, index_shift+n));
                }
                if(add_v_corrector)
                  sample_weight->at(index_shift+li) = 1.f - l2dist(trajectory[li].a, *all_pi, index_shift+li);
              } else if(offpolicy_strategy == 5) {
                double target_sum = 0;
                for (int n=trajectory.size()-1; n>=li; n--) {
                  target_sum += deltas[index_shift+n];
                  diff[index_shift+li] = target_sum;
                  target_sum *= this->gamma * lambda *
                                (1.f - std::min(l2dist(trajectory[n].a, *all_pi, index_shift+n),
                                                l2dist(trajectory[n].pure_a, *all_pi, index_shift+n)));
                }
                if(add_v_corrector)
                  sample_weight->at(index_shift+li) = 1.f - std::min(l2dist(trajectory[li].a, *all_pi, index_shift+li),
                                                      l2dist(trajectory[li].pure_a, *all_pi, index_shift+li));
              }

              li++;
            }
            ASSERT(diff[index_shift+trajectory.size() -1] == deltas[index_shift+trajectory.size() -1], "pb lambda ");
            index_shift += trajectory.size();
          }
          ASSERT(index_shift == all_size, "index pb");

          // comment following lines to compare with the other formula
          for (int i=0; i<all_size; i++)
            diff[i] = diff[i] + all_V->at(i);

//           bib::Logger::PRINT_ELEMENTS(*all_V, "all v ");
//           bib::Logger::PRINT_ELEMENTS(diff, "final diff ");
          if(!offpolicy_critic) {
            int size_last_traj = trajectories.back()->transitions->size();
            qnn->increase_batchsize(size_last_traj);

            std::vector<double> subdiff(size_last_traj);
            std::vector<double> subsensors(size_last_traj * this->get_state_size());
            std::vector<double> submotors(size_last_traj * this->nb_motors);

            const std::deque<sample>& trajectory = *trajectories.back()->transitions;

            li=0;
            for (auto it : trajectory) {
              std::copy(it.s.begin(), it.s.end(), subsensors.begin() + li * this->get_state_size());
              std::copy(it.a.begin(), it.a.end(), submotors.begin() + li * this->nb_motors);
              li++;
            }

            int index=size_last_traj-1;
            for(li=all_size - 1; li>=(all_size-size_last_traj); li--) {
              subdiff[index]=diff[li];
              index--;
            }
            qnn->learn_batch(subsensors, submotors, subdiff, stoch_iter_critic);

            qnn->increase_batchsize(all_size);
          } else if(offpolicy_strategy != 0 && add_v_corrector)
            qnn->learn_batch_lw(all_states, all_actions, diff, *sample_weight, stoch_iter_critic);
          else
            qnn->learn_batch(all_states, all_actions, diff, stoch_iter_critic);

          delete all_V;
        }

        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();

      delete all_next_actions;

      if(lambda >= 0.f)
        if(offpolicy_strategy != 0 && offpolicy_critic) {
          delete[] ptheta;
          delete all_pi;
          if(add_v_corrector)
            delete sample_weight;
        }
    }
  }

  void end_episode(bool learning) override {
    //     LOG_FILE("policy_exploration", ann->hash());
    if(!learning)
      return;

    uint all_size = alltransitions();
    if(all_size > 0) {
      qnn->increase_batchsize(all_size);
    }

    for(uint fi = 0 ; fi < number_global_fitted_iteration; fi++) {
      if(update_critic_first)
        update_critic();
      
      if(reset_actor){
        delete ann;
        ann = new NN(this->get_state_size(), *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type,
                     actor_output_layer_type,
                     batch_norm_actor, true);
      }

      if(!offpolicy_actor)
        actor_update_onpolicy();
      else
        actor_update_offpolicy();

      if(!update_critic_first) {
        if(all_size > 0 && !offpolicy_actor) {
          qnn->increase_batchsize(all_size);
        }
        update_critic();
      }
    }

    if(shuffle_buffer) {
      std::random_shuffle(trajectories.begin(), trajectories.end());
    }
  }

  void end_instance(bool learning) override {
    if(learning)
      episode++;
  }

  void actor_update_onpolicy() {
    const std::deque<sample>& trajectory = *trajectories.back()->transitions;
    qnn->increase_batchsize(trajectory.size());
    if (trajectory.size() > 0) {
      std::vector<bool> disable_back(trajectory.size() * this->nb_motors, false);
      const std::vector<bool> disable_back_ac(this->nb_motors, true);
//       std::vector<double> deltas_blob(trajectory.size() * this->nb_motors);
      std::vector<double> deltas(trajectory.size());

      std::vector<double> all_states(trajectory.size() * this->get_state_size());
      std::vector<double> all_next_states(trajectory.size() * this->get_state_size());
      std::vector<double> all_actions(trajectory.size() * this->nb_motors);
      uint li=0;
      double gamma_factor = 1.f;
      for (auto it : trajectory) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * this->get_state_size());
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * this->get_state_size());
        if(!determinist_update)
          std::copy(it.a.begin(), it.a.end(), all_actions.begin() + li * this->nb_motors);
        else if(gamma_corrector) {//store gamma values
          std::fill(all_actions.begin() + li * this->nb_motors, all_actions.begin() + (li+1) * this->nb_motors, gamma_factor);
          gamma_factor *= this->gamma;
        }
        li++;
      }
      
      ASSERT(li == trajectory.size(), "pb");

      uint n=0;
      ann->increase_batchsize(trajectory.size());
      if(!determinist_update){
        std::vector<double>* all_next_actions;
        all_next_actions = ann->computeOutBatch(all_next_states);
        //TODO:use_qmu -2
        std::vector<double>* all_nextV;
        if(use_qmu <= 0)
          all_nextV = qnn->computeOutVFBatch(all_next_states, *all_next_actions);
        else {
          std::vector<double> all_actions_mu(trajectory.size() * this->nb_motors * use_qmu);
          std::vector<double> all_states_tmp(trajectory.size() * this->get_state_size() * use_qmu);
          for(int j=0;j<use_qmu;j++){
            vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*all_next_actions, noise);
            std::copy(randomized_action->begin(), randomized_action->end(), all_actions_mu.begin() + j * trajectory.size() * this->nb_motors);
            std::copy(all_next_states.begin(), all_next_states.end(), all_states_tmp.begin() + j * trajectory.size() * this->get_state_size());
            delete randomized_action;
          }
          qnn->increase_batchsize(trajectory.size() * use_qmu);
          all_nextV = qnn->computeOutVFBatch(all_states_tmp, all_actions_mu);
          if(!use_qmu_actor)
            qnn->increase_batchsize(trajectory.size());
          double factor_ = use_qmu;
          for(uint j=0;j<trajectory.size();j++){
            double sum_ = 0.f;
            for(int k=0;k<use_qmu;k++)
              sum_ += all_nextV->at(k*trajectory.size()+j);
            all_nextV->at(j) = sum_ / factor_;
          }
        }
        std::vector<double>* all_mine;
        if(!use_qmu_actor){
          std::vector<double>* all_pi = ann->computeOutBatch(all_states);
          all_mine = qnn->computeOutVFBatch(all_states, *all_pi);
          delete all_pi;
        } else {
          std::vector<double> all_actions_mu(trajectory.size() * this->nb_motors * use_qmu);
          std::vector<double> all_states_tmp(trajectory.size() * this->get_state_size() * use_qmu);
          for(int j=0;j<use_qmu;j++){
            vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(all_actions, noise);
            std::copy(randomized_action->begin(), randomized_action->end(), all_actions_mu.begin() + j * trajectory.size() * this->nb_motors);
            std::copy(all_states.begin(), all_states.end(), all_states_tmp.begin() + j * trajectory.size() * this->get_state_size());
            delete randomized_action;
          }
          all_mine = qnn->computeOutVFBatch(all_states_tmp, all_actions_mu);
          qnn->increase_batchsize(trajectory.size());
          double factor_ = use_qmu;
          for(uint j=0;j<trajectory.size();j++){
            double sum_ = 0.f;
            for(int k=0;k<use_qmu;k++)
              sum_ += all_mine->at(k*trajectory.size()+j);
            all_mine->at(j) = sum_ / factor_;
          }
        }
        
        delete all_next_actions;

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
        ASSERT(li == trajectory.size(), "pb");

        if(gae) {
          //
          //        Simple computation for lambda return
          //
          std::vector<double> diff(trajectory.size());
          li=0;
          for (auto it : trajectory) {
            diff[li] = 0;
            for (uint j=li; j<trajectory.size(); j++)
              diff[li] += std::pow(this->gamma * lambda, j-li) * deltas[j];
            li++;
          }
          ASSERT(li == trajectory.size(), "pb");

          ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");
          li=0;
          for (auto it : trajectory) {
            //           diff[li] = diff[li] + all_V->at(li);
            deltas[li] = diff[li];
            ++li;
          }
          ASSERT(li == trajectory.size(), "pb");
        }

        li=0;
        for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
          sample sm = *it;

          if(deltas[li] > 0.) {
            n++;
          } else {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
          }
  //         std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
          li++;
        }
        
        ASSERT(li == trajectory.size(), "pb");
        delete all_nextV;
        delete all_mine;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      
      if(n > 0 || determinist_update) {
        for(uint sia = 0; sia < stoch_iter_actor; sia++) {
          //learn BN
          auto ac_out = ann->computeOutBatch(all_states);
          if(batch_norm_actor != 0 && batch_norm_testing){
            //re-compute ac_out with BN as testing
            ann_testing->copyParametersFrom(ann);
            ann_testing->increase_batchsize(trajectory.size());
            delete ac_out;
            ac_out = ann_testing->computeOutBatch(all_states);
            if(sia + 1 == stoch_iter_actor )
              ann_testing->increase_batchsize(1);
          }

          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          if(!determinist_update) {
            auto ac_diff = actor_actions_blob->mutable_cpu_diff();
            for(int i=0; i<actor_actions_blob->count(); i++) {
              if(disable_back[i]) {
                ac_diff[i] = 0.00000000f;
              } else {
                double x = all_actions[i] - ac_out->at(i);
                ac_diff[i] = -x;
              }
            }
          } else {
            qnn->ZeroGradParameters();
            ann->ZeroGradParameters();
            
            delete qnn->computeOutVFBatch(all_states, *ac_out);
            const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
            double* q_values_diff = q_values_blob->mutable_cpu_diff();
            for (int i=0; i<q_values_blob->count(); i++)
              q_values_diff[q_values_blob->offset(i,0,0,0)] = -1.0f;
            qnn->critic_backward();
            const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
            if(gamma_corrector){
              auto ac_diff_critic = critic_action_blob->cpu_diff();
              auto ac_diff = actor_actions_blob->mutable_cpu_diff();
              for(int i=0; i<actor_actions_blob->count(); i++)
                ac_diff[i] = ac_diff_critic[i] * all_actions[i];
            } else 
              actor_actions_blob->ShareDiff(*critic_action_blob);
          }
          ann->actor_backward();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
          delete ac_out;
        }
      }
    }
  }

  void actor_update_offpolicy() {
    int all_size = alltransitions();
    ann->increase_batchsize(all_size);

    std::vector<bool> disable_back(all_size * this->nb_motors, false);
    const std::vector<bool> disable_back_ac(this->nb_motors, true);
//     std::vector<double> deltas_blob(all_size * this->nb_motors);
    std::vector<double> deltas(all_size);

    std::vector<double> all_states(all_size * this->get_state_size());
    std::vector<double> all_next_states(all_size * this->get_state_size());
    std::vector<double> all_actions(all_size * this->nb_motors);

    int li=0;
    for(auto one_trajectory : trajectories) {
      const std::deque<sample>& trajectory = *one_trajectory->transitions;
      double gamma_factor = 1.f;
      for (auto it : trajectory) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * this->get_state_size());
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * this->get_state_size());
        if(!determinist_update)
          std::copy(it.a.begin(), it.a.end(), all_actions.begin() + li * this->nb_motors);
        else if(gamma_corrector) {//store gamma values
          std::fill(all_actions.begin() + li * this->nb_motors, all_actions.begin() + (li+1) * this->nb_motors, gamma_factor);
          gamma_factor *= this->gamma;
        }
        li++;
      }
    }

    if(!determinist_update) {
      std::vector<double>* all_next_actions ; 
      if(!ac_onpolnexta)
        all_next_actions = ann->computeOutBatch(all_next_states);
      else {
        all_next_actions = new std::vector<double>(all_size * this->nb_motors);
        li=0;
        for(auto one_trajectory : trajectories) {
          const std::deque<sample>& trajectory = *one_trajectory->transitions;
          
          bool skip_first=true;
          for (auto it : trajectory) {
            if(skip_first){
              skip_first=false;
              continue;
            }
            std::copy(it.a.begin(), it.a.end(), all_actions.begin() + li * this->nb_motors);
            li++;
          }
          
          vector<double>* next_action = ann_testing->computeOut(trajectory.back().s);
          std::copy(next_action->begin(), next_action->end(), all_actions.begin() + li * this->nb_motors);
          delete next_action;
          li++;
        }
      }
      std::vector<double>* all_nextV;
      if(use_qmu <= 0)
        all_nextV = qnn->computeOutVFBatch(all_next_states, *all_next_actions);
      else {
        std::vector<double> all_actions_mu(all_size * this->nb_motors * use_qmu);
        std::vector<double> all_states_tmp(all_size * this->get_state_size() * use_qmu);
        for(int j=0;j<use_qmu;j++){
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*all_next_actions, noise);
          std::copy(randomized_action->begin(), randomized_action->end(), all_actions_mu.begin() + j * all_size * this->nb_motors);
          std::copy(all_next_states.begin(), all_next_states.end(), all_states_tmp.begin() + j * all_size * this->get_state_size());
          delete randomized_action;
        }
        qnn->increase_batchsize(all_size * use_qmu);
        all_nextV = qnn->computeOutVFBatch(all_states_tmp, all_actions_mu);
        if(!use_qmu_actor)
          qnn->increase_batchsize(all_size);
        double factor_ = use_qmu;
        for(int j=0;j<all_size;j++){
          double sum_ = 0.f;
          for(int k=0;k<use_qmu;k++)
            sum_ += all_nextV->at(k*all_size+j);
          all_nextV->at(j) = sum_ / factor_;
        }
      }
      std::vector<double>* all_mine;
      if(!use_qmu_actor) {
        std::vector<double>* all_pi = ann->computeOutBatch(all_states);
        all_mine = qnn->computeOutVFBatch(all_states, *all_pi);
        delete all_pi;
      } else {
        std::vector<double> all_actions_mu(all_size * this->nb_motors * use_qmu);
        std::vector<double> all_states_tmp(all_size * this->get_state_size() * use_qmu);
        for(int j=0;j<use_qmu;j++){
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(all_actions, noise);
          std::copy(randomized_action->begin(), randomized_action->end(), all_actions_mu.begin() + j * all_size * this->nb_motors);
          std::copy(all_states.begin(), all_states.end(), all_states_tmp.begin() + j * all_size * this->get_state_size());
          delete randomized_action;
        }
        all_mine = qnn->computeOutVFBatch(all_states_tmp, all_actions_mu);
        qnn->increase_batchsize(all_size);
        double factor_ = use_qmu;
        for(int j=0;j<all_size;j++){
          double sum_ = 0.f;
          for(int k=0;k<use_qmu;k++)
            sum_ += all_mine->at(k*all_size+j);
          all_mine->at(j) = sum_ / factor_;
        }
      }
      
      delete all_next_actions;

      li=0;
      for(auto one_trajectory : trajectories) {
        const std::deque<sample>& trajectory = *one_trajectory->transitions;
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
      }
      
      delete all_nextV;
      delete all_mine;
    }

    decltype(ann->computeOutBatch(all_states)) all_pi;
    if(gae && !determinist_update) {
      //
      //        Simple computation for lambda return
      //
      std::vector<double> diff(all_size);
      int index_shift=0;
      double* ptheta;
      double max_ptheta;
      if(offpolicy_strategy != 0) {
        all_pi = ann->computeOutBatch(all_states);
        ptheta = new double[all_size];

        uint i=0;
        if(offpolicy_strategy >= 1 && offpolicy_strategy <= 3) {
          for(auto one_trajectory : trajectories) {
            const std::deque<sample>& trajectory = *one_trajectory->transitions;
            for (auto it : trajectory) {
              double p0 = 1.f;
              for(uint j=0; j < this->nb_motors; j++)
                p0 *= bib::Proba<double>::truncatedGaussianDensity(it.a[j], all_pi->at(i*this->nb_motors+j), noise);

              ptheta[i] = p0;
              i++;
            }
          }
          max_ptheta = *std::max_element(ptheta, ptheta+all_size);
        }
      }
      for(auto one_trajectory : trajectories) {
        li=0;
        const std::deque<sample>& trajectory = *one_trajectory->transitions;
        for (auto it : trajectory) {
          diff[index_shift+li] = 0;
          if(offpolicy_strategy == 0) {
            for (uint n=li; n<trajectory.size(); n++)
              diff[index_shift+li] += std::pow(this->gamma * lambda, n-li) * deltas[index_shift+n];
          } else if(offpolicy_strategy == 1) {
            double target_sum = 0;
            for (int n=trajectory.size()-1; n>=li; n--) {
              target_sum += deltas[index_shift+n];
              diff[index_shift+li] = target_sum;
              target_sum *= this->gamma * lambda * (ptheta[index_shift+n]/max_ptheta);
            }
          } else if(offpolicy_strategy == 2) {
            double target_sum = 0;
            for (int n=trajectory.size()-1; n>=li; n--) {
              target_sum += deltas[index_shift+n];
              diff[index_shift+li] = target_sum;
              target_sum *= this->gamma * lambda * (ptheta[index_shift+n]/trajectory[n].dpmu);
            }
          } else if(offpolicy_strategy == 3) {
            double target_sum = 0;
            for (int n=trajectory.size()-1; n>=li; n--) {
              target_sum += deltas[index_shift+n];
              diff[index_shift+li] = target_sum;
              target_sum *= this->gamma * lambda * std::min((double)1.f,ptheta[index_shift+n]/trajectory[n].dpmu);
            }
          } else if(offpolicy_strategy == 4) {
            double target_sum = 0;
            for (int n=trajectory.size()-1; n>=li; n--) {
              target_sum += deltas[index_shift+n];
              diff[index_shift+li] = target_sum;
              target_sum *= this->gamma * lambda * (1.f - l2dist(trajectory[n].a, *all_pi, index_shift+n));
            }
          } else if(offpolicy_strategy == 5) {
            double target_sum = 0;
            for (int n=trajectory.size()-1; n>=li; n--) {
              target_sum += deltas[index_shift+n];
              diff[index_shift+li] = target_sum;
              target_sum *= this->gamma * lambda *
                            (1.f - std::min(l2dist(trajectory[n].a, *all_pi, index_shift+n),
                                            l2dist(trajectory[n].pure_a, *all_pi, index_shift+n)));
            }
          }
          li++;
        }
        ASSERT(diff[index_shift+trajectory.size() -1] == deltas[index_shift+trajectory.size() -1], "pb lambda");
        index_shift += trajectory.size();
      }

      if(offpolicy_strategy != 0) {
        //don't delete all_pi it'll be used later
        delete [] ptheta;
      }

      for (li=0; li < all_size; li++) {
        //           diff[li] = diff[li] + all_V->at(li);
        deltas[li] = diff[li];
      }
    }

    uint n=0;
    if(!determinist_update){
      li=0;
      for(auto one_trajectory : trajectories) {
        const std::deque<sample>& trajectory = *one_trajectory->transitions;
        for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
          if(deltas[li] > 0.) {
            n++;
          } else {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
          }
  //         std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
          li++;
        }
      }
    }

    ratio_valid_advantage = ((float)n) / ((float) all_size);
    
    if(n > 0 || determinist_update) {
      for(uint sia = 0; sia < stoch_iter_actor; sia++) {
        decltype(ann->computeOutBatch(all_states)) ac_out;
        //learn BN
        if(stoch_iter_actor == 1 && gae && offpolicy_strategy != 0)
          ac_out = all_pi; // I already computeOutBatch sensors
        else {
          ac_out = ann->computeOutBatch(all_states);
        }
        
        if(batch_norm_actor != 0 && batch_norm_testing){
          //re-compute ac_out with BN as testing
          ann_testing->copyParametersFrom(ann);
          ann_testing->increase_batchsize(all_size);
          delete ac_out;
          ac_out = ann_testing->computeOutBatch(all_states);
          if(sia + 1 == stoch_iter_actor )
            ann_testing->increase_batchsize(1);
        }

        const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
        if(!determinist_update) {
          auto ac_diff = actor_actions_blob->mutable_cpu_diff();
          for(int i=0; i<actor_actions_blob->count(); i++) {
            if(disable_back[i]) {
              ac_diff[i] = 0.00000000f;
            } else {
              double x = all_actions[i] - ac_out->at(i);
              ac_diff[i] = -x;
            }
          }
        } else {
          qnn->ZeroGradParameters();
          ann->ZeroGradParameters();

          delete qnn->computeOutVFBatch(all_states, *ac_out);
          const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
          double* q_values_diff = q_values_blob->mutable_cpu_diff();
          for (int i=0; i<q_values_blob->count(); i++)
            q_values_diff[q_values_blob->offset(i,0,0,0)] = -1.0f;
          qnn->critic_backward();
          const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
          if(gamma_corrector){
            auto ac_diff_critic = critic_action_blob->cpu_diff();
            auto ac_diff = actor_actions_blob->mutable_cpu_diff();
            for(int i=0; i<actor_actions_blob->count(); i++)
              ac_diff[i] = ac_diff_critic[i] * all_actions[i];
          } else 
            actor_actions_blob->ShareDiff(*critic_action_blob);
        }
        ann->actor_backward();
        ann->getSolver()->ApplyUpdate();
        ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
        delete ac_out;
      }
    } else if (gae && offpolicy_strategy != 0)
      delete all_pi;

  }

  void save(const std::string& path, bool, bool) override {
    ann->save(path+".actor");
    qnn->save(path+".critic");
  }

  void save_run() override {
    ann->save("continue.actor");
    qnn->save("continue.critic");
    struct algo_state st = {episode};
    bib::XMLEngine::save(st, "algo_state", "continue.algo_state.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    qnn->load(path+".critic");
  }

  void load_previous_run() override {
    ann->load("continue.actor");
    qnn->load("continue.critic");
    auto p3 = bib::XMLEngine::load<struct algo_state>("algo_state", "continue.algo_state.data");
    episode = p3->episode;
    delete p3;
  }

 protected:

  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << qnn->error() << " " << noise << " " << alltransitions() <<
          " " << ratio_valid_advantage;
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
        this->sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << qnn->error() << " " << alltransitions() << " " << ratio_valid_advantage;
  }

  uint alltransitions() const {
    uint count=0;
    for (auto it : trajectories)
      count += it->transitions->size();
    return count;
  }

  double sign(double x) {
    if(x>=0)
      return 1.f;
    return -1.f;
  }

  double l2dist(const std::vector<double>& a, const std::vector<double>& b, const uint start) const {
    double r = 0.f;
    for(uint i=0; i<a.size(); i++) {
      double d = (a[i] - b[start+i]);
      r += d*d;
    }
    return sqrt(r)/(2.f *((double) a.size()));
  }

 private:
  uint episode = 0;

  double noise;
  bool gaussian_policy, vnn_from_scratch, update_critic_first,
       gae, add_v_corrector, offpolicy_actor, determinist_update;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type,
       hidden_layer_type, max_trajectory, offpolicy_strategy;
  double lambda;
  uint number_global_fitted_iteration;
  bool offpolicy_critic, shuffle_buffer, gamma_corrector, batch_norm_testing;
  int use_qmu;
  bool use_qmu_actor, ac_onpolnexta, reset_actor;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<std::shared_ptr<trajectory>> trajectories;

  NN* ann;
  NN* qnn;
  NN* ann_testing;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference
  float ratio_valid_advantage;

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


