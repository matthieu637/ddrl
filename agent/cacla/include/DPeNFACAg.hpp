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
#include "bib/MetropolisHasting.hpp"
#include "bib/XMLEngine.hpp"
#include "bib/IniParser.hpp"
#include "nn/MLP.hpp"

#ifndef SAASRG_SAMPLE
#define SAASRG_SAMPLE
typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;

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
    delete advnn;
    
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
    disable_trust_region = pt->get<bool>("agent.disable_trust_region");
    disable_cac                 = pt->get<bool>("agent.disable_cac");
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
    
    std::vector<uint> linear(0);
    advnn = new NN(ann->number_of_parameters(true), ann->number_of_parameters(true), linear, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, false, momentum);
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    if(batch_norm_critic != 0)
      vnn_testing = new NN(*vnn, false, ::caffe::Phase::TEST);
    
    bestever_score = std::numeric_limits<double>::lowest();
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    step = 0;
    if(gaussian_policy == 2)
      oun->reset();
    
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
          vnn->learn_batch(all_states, empty_action, diff, stoch_iter_critic);

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
    
    if(update_critic_first)
      update_critic();

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
      
      //learn compta critic there
      std::vector<double> mu_minus_a(trajectory.size() * this->nb_motors);
      li=0;
      for(int i=0;i< trajectory.size() ;i++){
          for(int j=0;j< this->nb_motors ;j++){
            mu_minus_a[li] = trajectory[i].pure_a[j] - trajectory[i].a[j];
            li++;
          }
      }
      auto lann = ann;
      if(batch_norm_actor != 0)
          lann = ann_testing;
      lann->increase_batchsize(1);
      std::vector<double> features(lann->number_of_parameters(true) * trajectory.size());
      for (int j = 0;j < trajectory.size();j++) {
        lann->ZeroGradParameters();
        delete lann->computeOutBatch(trajectory[j].s);
        const auto actor_actions_blob = lann->getNN()->blob_by_name(MLP::actions_blob_name);
        auto ac_diff = actor_actions_blob->mutable_cpu_diff();
        for(int i=0; i<actor_actions_blob->count(); i++){
//             if(deltas[j] > 0)
                ac_diff[i] = trajectory[j].pure_a[i] - trajectory[j].a[i];
//             else 
//                 ac_diff[i] = 0.0000f;
        }
        lann->actor_backward();
        double* weights_diff = new double[lann->number_of_parameters(true)];
        lann->copyDiffTo(weights_diff, true);
        std::copy(weights_diff, weights_diff + lann->number_of_parameters(true), features.begin() + j * lann->number_of_parameters(true));
        delete[] weights_diff;
      }
      
      advnn->increase_batchsize(trajectory.size());
      advnn->learn_batch(features, empty_action, deltas, stoch_iter_critic*50);
      
      uint n=0;
      posdelta_mean=0.f;
      std::vector<double> sensors(trajectory.size() * nb_sensors);
      std::vector<bool> disable_back(trajectory.size() * this->nb_motors, false);
      std::vector<double> deltas_blob(trajectory.size() * this->nb_motors, 1.f);

      li=0;
      //cacla cost
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        std::copy(it->s.begin(), it->s.end(), sensors.begin() + li * nb_sensors);
        if(deltas[li] > 0.) {
          posdelta_mean += deltas[li];
          n++;
        } else {
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), disable_back.begin() + li * this->nb_motors);
        }
        if(!disable_cac)
            std::fill(deltas_blob.begin() + li * this->nb_motors, deltas_blob.begin() + (li+1) * this->nb_motors, deltas[li]);
        li++;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      posdelta_mean = posdelta_mean / ((float) trajectory.size());

//       Vanilla compatible gradient
//       if(n > 0) {
//           ann->increase_batchsize(trajectory.size());
//           ann->ZeroGradParameters();
// 
//           const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
//           auto ac_diff = actor_actions_blob->mutable_cpu_diff();
//           
//           auto outadv = advnn->computeOutVFBatch(features, empty_action);
//           delete ann->computeOutBatch(sensors);
//           for(int i=0; i<actor_actions_blob->count(); i++) {
//             if(disable_back[i])
//                 ac_diff[i] = 0.00000000f;
//             else
//                 ac_diff[i] = mu_minus_a[i]*outadv->at(i/this->nb_motors);
//           }
// //           bib::Logger::PRINT_ELEMENTS(ac_diff, actor_actions_blob->count());
//           ann->actor_backward();
//           ann->updateFisher(n);
//           ann->regularize();
//           ann->getSolver()->ApplyUpdate();
//           ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
//           
//           delete outadv;
//       }
      
//       Natural grad
      ann->ZeroGradParameters();
      double* weights_diff = new double[ann->number_of_parameters(true)];
      advnn->copyWeightsTo(weights_diff, true);
      ann->copyDiffFrom(weights_diff, true);
      delete[] weights_diff;
      ann->updateFisher(n);
      ann->regularize();
      ann->getSolver()->ApplyUpdate();
      ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
      
      
      mean_beta /= (double) number_effective_actor_update;

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
        " " << std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << nb_sample_update <<
          " " << std::setprecision(3) << ratio_valid_advantage << " " << vnn->weight_l1_norm() << " " << ann->weight_l1_norm(true);
  }

//clear all; close all; wndw = 10; X=load('0.learning.data'); X=filter(ones(wndw,1)/wndw, 1, X); startx=0; starty=800; width=350; height=350; figure('position',[startx,starty,width,height]); plot(X(:,3), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('sum rewards', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]); plot(X(:,9), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('beta', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,8), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('valid adv', "fontsize", 16); ylim([0, 1]); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,11), "linewidth", 2); hold on; plot(X(:,12), "linewidth", 2, "color", "red"); legend("critic", "actor"); xlabel('learning episode', "fontsize", 16); ylabel('||\theta||_1', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,10), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('||\mu_{old}-\mu||_2', "fontsize", 16); startx+=width; figure('position',[startx,starty,width,height]) ; plot(X(:,14), "linewidth", 2); xlabel('learning episode', "fontsize", 16); ylabel('effective actor. upd.', "fontsize", 16); 
  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) << this->sum_weighted_reward/this->gamma << " " << 
    this->sum_reward << " " << std::setw(8) << std::fixed << std::setprecision(5) << vnn->error() << " " << 
    nb_sample_update << " " << std::setprecision(3) << ratio_valid_advantage << " " << std::setprecision(10) << 
    mean_beta << " " << conserved_l2dist << " " << std::setprecision(3) << vnn->weight_l1_norm() << " " << 
    ann->weight_l1_norm(true) << " " << std::setprecision(6)  << posdelta_mean << " " << number_effective_actor_update;
  }
  
 private:
  uint nb_sensors;
  uint episode = 1;
  uint step = 0;

  double noise, noise2, noise3;
  uint gaussian_policy;
  bool update_critic_first, gae, ignore_poss_ac, conserve_beta, disable_trust_region, disable_cac;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, beta_target;
  double conserved_beta= 0.0001f;
  double mean_beta= 0.f;
  double conserved_l2dist= 0.f;
  int number_effective_actor_update = 0;

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
  NN* advnn;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  float ratio_valid_advantage=0;
  int nb_sample_update = 0;
  double posdelta_mean = 0;
  
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

