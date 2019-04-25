#ifndef PENNFAC_HPP
#define PENNFAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include "bib/OrnsteinUhlenbeckNoise.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "nn/MLP.hpp"
#include "nn/DODevMLP.hpp"

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
#endif

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
    delete ann_testing_noisy;
    
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action_pure;
    vector<double>* next_action;
    if(learning){
        next_action = ann_testing_noisy->computeOut(sensors);
        next_action_pure = ann_testing->computeOut(sensors);
    } else {
        next_action = ann_testing->computeOut(sensors);
    }
    
    if (last_action.get() != nullptr && learning)
      trajectory.push_back( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached});

    if(learning)
        last_pure_action.reset(next_action_pure);
    last_action.reset(next_action);

    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    step++;
    
    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
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
    beta_target   = pt->get<double>("agent.beta_target");
    ignore_poss_ac        = pt->get<bool>("agent.ignore_poss_ac");
    adaptive_noise        = pt->get<bool>("agent.adaptive_noise");
    gae                     = false;
    update_each_episode = 1;
    effective_noise = noise;
    
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
    if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);
    
    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, false, momentum);
    if(std::is_same<NN, DODevMLP>::value)
      vnn->exploit(pt, ann);
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    ann_testing_noisy = new NN(*ann, false, ::caffe::Phase::TEST);
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
    
    ratio_valid_advantage = -1;
    
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
    
    std::vector<double> embedded(weights, weights + ann->number_of_parameters(false));
    std::vector<double>* noisy_weights = bib::Proba<double>::multidimentionnalGaussian(embedded, effective_noise);
    ann_testing_noisy->copyWeightsFrom(noisy_weights->data(), false);
    
    delete[] weights;
    delete noisy_weights;
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
          
// //           comment following lines to compare with the other formula
          li=0;
          for (auto it : trajectory){
            diff[li] = diff[li] + all_V->at(li);
            ++li;
          }
          
//          bib::Logger::PRINT_ELEMENTS(diff, "target ");
          V_pi_s0 = diff[0];
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
    
    return V_pi_s0;
  }

  void end_episode(bool learning) override {
//     LOG_FILE("policy_exploration", ann->hash());
    if(!learning){
      if(ann->ewc_best_method() >= 4){
        ann->update_best_param_previous_task(this->sum_weighted_reward);
        vnn->update_best_param_previous_task(this->sum_weighted_reward);
      }
      return;
    }
    
    //learning phase
    if(ann->ewc_best_method() <= 3){
      ann->update_best_param_previous_task(this->sum_weighted_reward);
      vnn->update_best_param_previous_task(this->sum_weighted_reward);
    }
    
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
      
      uint n=0;
      std::vector<double> sensors(2*trajectory.size() * nb_sensors);
      std::vector<double> actions(2*trajectory.size() * this->nb_motors);
      std::vector<bool> disable_back(2*trajectory.size() * this->nb_motors, false);
      std::vector<double> deltas_blob(2*trajectory.size() * this->nb_motors);

      li=0;
      //cacla cost
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        std::copy(it->a.begin(), it->a.end(), actions.begin() + li * this->nb_motors);
        if(deltas[li] > 0.) {
          n++;
        } else {
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
        li++;
      }
      //penalty cost
      int li2=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        std::copy(it->pure_a.begin(), it->pure_a.end(), actions.begin() + li * this->nb_motors);
        if(ignore_poss_ac && deltas[li2] > 0.) {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li2]);
        li++;
        li2++;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      
      double beta=1.f;
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
          l2distance = std::sqrt(l2distance)/((double) trajectory.size()*this->nb_motors);
          if (l2distance < beta_target/1.5)
              beta = beta/2.;
          else if (l2distance > beta_target*1.5)
              beta = beta*2.;
          else if (sia > 0)
              break;
          
          if (sia == 0 && adaptive_noise){
              if (l2distance < effective_noise)
                  effective_noise = 1.01f * effective_noise;
              else
                  effective_noise = (1.f/1.01f) * effective_noise;
          }
          
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
                    ac_diff[i] = -x * beta * deltas_blob[i];
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward/this->gamma << " " << this->sum_reward << 
        " " << std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << effective_noise << " " << nb_sample_update <<
          " " << std::setprecision(3) << ratio_valid_advantage << " " << vnn->weight_l1_norm() << " " << ann->weight_l1_norm();
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
    this->sum_weighted_reward/this->gamma << " " << this->sum_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << nb_sample_update <<
        " " << std::setprecision(3) << ratio_valid_advantage ;
  }
  
 private:
  uint nb_sensors;
  uint episode = 1;
  uint step = 0;

  double noise, effective_noise;
  bool vnn_from_scratch, update_critic_first, gae, ignore_poss_ac, adaptive_noise;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, beta_target;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<sample> trajectory;
  std::deque<int> trajectory_end_points;

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* ann_testing_noisy;
  NN* vnn_testing;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  float ratio_valid_advantage=0;
  int nb_sample_update = 0;
  
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

