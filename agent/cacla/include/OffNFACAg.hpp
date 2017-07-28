#ifndef OFFNFACAG_HPP
#define OFFNFACAG_HPP

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
class OffNFACAg : public arch::AACAgent<NN, arch::AgentProgOptions> {
 public:
  typedef NN PolicyImpl;

  OffNFACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<NN, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors), empty_action(0) {

  }

  virtual ~OffNFACAg() {
    delete vnn;
    delete ann;

    delete ann_testing;
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) {

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
    max_trajectory          = pt->get<uint>("agent.max_trajectory");
    offpolicy_strategy      = pt->get<uint>("agent.offpolicy_strategy");
    add_v_corrector         = pt->get<bool>("agent.add_v_corrector");
    corrected_update_ac     = false;
    gae                     = false;
    try {
      corrected_update_ac   = pt->get<bool>("agent.corrected_update_ac");
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

    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, add_v_corrector);
    if(std::is_same<NN, DODevMLP>::value)
      vnn->exploit(pt, ann);

    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    if(batch_norm_critic != 0)
      vnn_testing = new NN(*vnn, false, ::caffe::Phase::TEST);

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

    if(std::is_same<NN, DODevMLP>::value) {
      static_cast<DODevMLP *>(vnn)->inform(episode);
      static_cast<DODevMLP *>(ann)->inform(episode);
      static_cast<DODevMLP *>(ann_testing)->inform(episode);
    }

    double* weights = new double[ann->number_of_parameters(false)];
    ann->copyWeightsTo(weights, false);
    ann_testing->copyWeightsFrom(weights, false);
    delete[] weights;
  }

  void update_critic() {
    int all_size = alltransitions();
    if (all_size > 0) {
      //remove trace of old policy
      auto iter = [&]() {
        std::vector<double> all_states(all_size * nb_sensors);
        std::vector<double> all_next_states(all_size * nb_sensors);
        std::vector<double> v_target(all_size);

        int li=0;
        for(auto one_trajectory : trajectories) {
          const std::deque<sample>& trajectory = *one_trajectory->transitions;

          for (auto it : trajectory) {
            std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
            std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
            li++;
          }
        }
        ASSERT(li == all_size, "pb");

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
          delete vnn;
          vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, all_size, -1, hidden_layer_type,
                       batch_norm_critic, add_v_corrector);
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

            const auto v_values_blob = vnn->getNN()->blob_by_name(MLP::q_values_blob_name);
            double* v_values_diff = v_values_blob->mutable_cpu_diff();

            double s = all_size;
            for (int i=0; i<all_size; i++)
              v_values_diff[i] = (all_V->at(i)-v_target[i])/s;
            vnn->critic_backward();
            vnn->getSolver()->ApplyUpdate();
            vnn->getSolver()->set_iter(vnn->getSolver()->iter() + 1);
            delete all_V;
          }
        } else {
          auto all_V = vnn->computeOutVFBatch(all_states, empty_action);
          std::vector<double> deltas(all_size);
          //
          //        Simple computation for lambda return
          //
          for (int i=0; i<all_size; i++)
            deltas[i] = v_target[i] - all_V->at(i);

          decltype(ann->computeOutBatch(all_states)) all_pi;
          double* ptheta;
          double max_ptheta;
          std::vector<double>* sample_weight;
          if(offpolicy_strategy != 0) {
            ann->increase_batchsize(all_size);
            all_pi = ann->computeOutBatch(all_states);
            ptheta = new double[all_size];

            uint i=0;
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
            
            if(add_v_corrector)
              sample_weight = new std::vector<double>(all_size);
          }

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
              if(offpolicy_strategy == 0) {
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
                  sample_weight->at(index_shift+li) = std::min((double)1.f,ptheta[index_shift+li]/trajectory[li].dpmu);
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
          if(offpolicy_strategy != 0 && add_v_corrector)
            vnn->learn_batch_lw(all_states, empty_action, diff, *sample_weight, stoch_iter_critic);
          else
            vnn->learn_batch(all_states, empty_action, diff, stoch_iter_critic);
          
          if(offpolicy_strategy != 0) {
            delete[] ptheta;
            delete all_pi;
            if(add_v_corrector)
              delete sample_weight;
          }
          
          delete all_V;
        }

        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();
    }
  }

  void end_episode(bool learning) override {
    //     LOG_FILE("policy_exploration", ann->hash());
    if(!learning)
      return;

    uint all_size = alltransitions();
    if(all_size > 0) {
      vnn->increase_batchsize(all_size);
      if(batch_norm_critic != 0)
        vnn_testing->increase_batchsize(all_size);
    }

    if(update_critic_first)
      update_critic();


    const std::deque<sample>& trajectory = *trajectories.back()->transitions;
    vnn->increase_batchsize(trajectory.size());
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
              if(!corrected_update_ac)
                ac_diff[i] = -x;
              else {
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
      if(all_size > 0) {
        vnn->increase_batchsize(all_size);
        if(batch_norm_critic != 0)
          vnn_testing->increase_batchsize(all_size);
      }
      update_critic();
    }
  }

  void end_instance(bool learning) override {
    if(learning)
      episode++;
  }

  void save(const std::string& path, bool) override {
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
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << alltransitions();
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
        this->sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << alltransitions() ;
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
  uint nb_sensors;
  uint episode = 0;

  double noise;
  bool gaussian_policy, vnn_from_scratch, update_critic_first,
        update_delta_neg, corrected_update_ac, gae, add_v_corrector;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type,
       hidden_layer_type, max_trajectory, offpolicy_strategy;
  double lambda, corrected_update_ac_factor;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<std::shared_ptr<trajectory>> trajectories;

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* vnn_testing;

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

