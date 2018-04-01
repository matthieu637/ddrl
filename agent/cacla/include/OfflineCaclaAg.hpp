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
#include "bib/OrnsteinUhlenbeckNoise.hpp"
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
class OfflineCaclaAg : public arch::AACAgent<NN, arch::AgentProgOptions> {
 public:
  typedef NN PolicyImpl;

  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<NN, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors), empty_action(0) {

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


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
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
    corrected_update_ac     = false;
    gae                     = false;
    inverting_gradient      = false;
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
    
    try {
      corrected_update_ac   = pt->get<bool>("agent.corrected_update_ac");
    } catch(boost::exception const& ) {
    }
    try {
      inverting_gradient   = pt->get<bool>("agent.inverting_gradient");
    } catch(boost::exception const& ) {
    }
    if(corrected_update_ac){
      try {
        corrected_update_ac_factor   = pt->get<double>("agent.corrected_update_ac_factor");
      } catch(boost::exception const& ) {
      }
    }
    
    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");
    
    if(lambda >=0. && batch_norm_critic != 0){
      LOG_DEBUG("to be done!");
      exit(1);
    }
    
    ann = new NN(nb_sensors, *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type, batch_norm_actor, true);
    if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);
    
    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic);
    if(std::is_same<NN, DODevMLP>::value)
      vnn->exploit(pt, ann);
    
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
    delete[] weights;
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
          vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, trajectory.size(), -1, hidden_layer_type, batch_norm_critic);
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
          li=0;
          for (auto it : trajectory){
            deltas[li] = v_target[li] - all_V->at(li);
            ++li;
          }
          
          std::vector<double> diff(trajectory.size());
          li=0;
          for (auto it : trajectory){
            diff[li] = 0;
            for (uint n=li;n<trajectory.size();n++)
              diff[li] += std::pow(this->gamma * lambda, n-li) * deltas[n];
            li++;
          }
          ASSERT(diff[trajectory.size() -1] == deltas[trajectory.size() -1], "pb lambda");
          
// //           comment following lines to compare with the other formula
          li=0;
          for (auto it : trajectory){
            diff[li] = diff[li] + all_V->at(li);
            ++li;
          }
          
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
    
    if(ann->ewc_best_method() < 3){
      ann->update_best_param_previous_task(this->sum_weighted_reward);
      vnn->update_best_param_previous_task(this->sum_weighted_reward);
    }    
      
    if(episode % update_each_episode != 0)
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
        li=0;
        for (auto it : trajectory){
          diff[li] = 0;
          for (uint n=li;n<trajectory.size();n++)
            diff[li] += std::pow(this->gamma * lambda, n-li) * deltas[n];
          li++;
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
      li=0;
//       double dmax = std::max((double)0.f, *std::max_element(deltas.begin(), deltas.end()));
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

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      
      if(n > 0) {
        for(uint sia = 0; sia < stoch_iter_actor; sia++){
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
          ann->ZeroGradParameters();
          
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          auto ac_diff = actor_actions_blob->mutable_cpu_diff();
          for(int i=0; i<actor_actions_blob->count(); i++){
            if(disable_back[i]){
              ac_diff[i] = 0.00000000f;
            } else {
              double x = actions[i] - ac_out->at(i);
              if(!corrected_update_ac){
                ac_diff[i] = -x;
                if(inverting_gradient){
                  const double min_ = -1.0; 
                  const double max_ = 1.0;
                  
                  if (ac_diff[i] < 0)
                    ac_diff[i] *= (max_ - ac_out->at(i)) / (max_ - min_);
                  else if (ac_diff[i] > 0)
                    ac_diff[i] *= (ac_out->at(i) - min_) / (max_ - min_);
                }
              } else {
                if(bib::Utils::rand01() < 0.4)
                  ac_diff[i] = -x ;//* (corrected_update_ac_factor + fabs(deltas_blob[i]));
                else
                  ac_diff[i] = 0.00000000f;
              }
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << nb_sample_update <<
          " " << std::setprecision(3) << ratio_valid_advantage ;
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) <<
    this->sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << nb_sample_update <<
        " " << std::setprecision(3) << ratio_valid_advantage ;
  }
  
  double sign(double x){
    if(x>=0)
      return 1.f;
    return -1.f;
  }

 private:
  uint nb_sensors;
  uint episode = 0;
  uint step = 0;

  double noise, noise2, noise3;
  uint gaussian_policy;
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

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* vnn_testing;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
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

