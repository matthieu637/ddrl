#ifndef PENNFAC_HPP
#define PENNFAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <caffe/util/math_functions.hpp>

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include "bib/OrnsteinUhlenbeckNoise.hpp"
#include "bib/MetropolisHasting.hpp"
#include "bib/XMLEngine.hpp"
#include "bib/IniParser.hpp"
#include "bib/OnlineNormalizer.hpp"
#include "nn/MLP.hpp"

#ifdef PARALLEL_INTERACTION
#include <mpi.h>
#include <boost/mpi.hpp>
#endif

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

} sample;
#endif

template<typename NN = MLP>
class OfflineCaclaAg : public arch::AACAgent<NN, arch::AgentGPUProgOptions> {
 public:
  typedef NN PolicyImpl;
  friend class FusionOOAg;

  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<NN, arch::AgentGPUProgOptions>(_nb_motors, _nb_sensors), nb_sensors(_nb_sensors), empty_action(), last_state(_nb_sensors, 0.f) {
      
  }

  virtual ~OfflineCaclaAg() {
    delete qnn;
    delete vnn;
    delete ann;
    
    delete ann_testing;
    if(batch_norm_actor != 0)
      delete ann_testing_blob;
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
    delete hidden_unit_q;
    
    if(normalizer_type > 0)
      delete normalizer;

    if(normalizer_action_type > 0)
      delete normalizer_action;
    
    if(oun == nullptr)
      delete oun;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors_,
                                  bool learning, bool goal_reached, bool) override {

    // protect batch norm from testing data and poor data
    vector<double>* next_action = nullptr;
    if(normalizer_type == 0) {
      next_action = ann_testing->computeOut(sensors_);
    } else {
      std::vector<double> sensors(nb_sensors);
      if (normalizer_type == 1)
        normalizer->transform(sensors, sensors_, false);
      else if (normalizer_type == 2)
        normalizer->transform_with_clip(sensors, sensors_, false);
      else
        normalizer->transform_with_double_clip(sensors, sensors_, false);
      next_action = ann_testing->computeOut(sensors);
    }
    
    if (last_action.get() != nullptr && learning)
      trajectory.push_back( {last_state, *last_pure_action, *last_action, sensors_, reward, goal_reached});

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

    std::copy(sensors_.begin(), sensors_.end(), last_state.begin());
    step++;
    
    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
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
    lambdaQ                 = pt->get<double>("agent.lambdaQ");
    momentum                = pt->get<uint>("agent.momentum");
    beta_target             = pt->get<double>("agent.beta_target");
    disable_cac             = pt->get<bool>("agent.disable_cac");
    learn_q_mu              = pt->get<bool>("agent.learn_q_mu");
    double decay_q          = pt->get<double>("agent.decay_q");
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
#ifdef PARALLEL_INTERACTION
      if (update_each_episode % (world.size() - 1) != 0) {
        LOG_ERROR("update_each_episode must be a multiple of (number of worker - 1)");
        exit(1);
      }
      update_each_episode = update_each_episode / (world.size() - 1);
#endif
    } catch(boost::exception const& ) {
    }
    
    normalizer_type = 0;
    try {
      normalizer_type     = pt->get<uint>("agent.normalizer_type");
    } catch(boost::exception const& ) {
    }
    if(normalizer_type > 0)
      normalizer = new bib::OnlineNormalizer(this->nb_sensors);

    normalizer_action_type = 0;
    try {
      normalizer_action_type     = pt->get<uint>("agent.normalizer_action_type");
    } catch(boost::exception const& ) {
    }
    if(normalizer_action_type > 0)
      normalizer_action = new bib::OnlineNormalizer(this->nb_motors);

    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");
    
    if(lambda >=0. && batch_norm_critic != 0 && stoch_iter_critic > 1){
      LOG_DEBUG("to be done!");
      exit(1);
    }

#ifdef CAFFE_CPU_ONLY
    LOG_INFO("CPU mode");
    (void) command_args;
#else
    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0
#ifdef PARALLEL_INTERACTION
      || world.rank() != 0
#endif
     ) 
    {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
      LOG_INFO("CPU mode");
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
      caffe::Caffe::SetDevice((*command_args)["gpu"].as<uint>());
      LOG_INFO("GPU mode");
    }
#endif
  
    ann = new NN(nb_sensors, *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type, batch_norm_actor, true, momentum);

    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, false, momentum);

    qnn = new NN(nb_sensors + this->nb_motors, nb_sensors, *hidden_unit_q, alpha_v, 1, decay_q, hidden_layer_type, batch_norm_critic, false, momentum);

#ifdef PARALLEL_INTERACTION
      std::vector<double> weights(ann->number_of_parameters(false), 0.f);
      if (world.rank() == 0)
          ann->copyWeightsTo(weights.data(), false);

      broadcast(world, weights, 0);
      if (world.rank() != 0)
          ann->copyWeightsFrom(weights.data(), false);
#endif

    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    
    if(batch_norm_actor != 0)
      ann_testing_blob = new NN(*ann, false, ::caffe::Phase::TEST);
    if(batch_norm_critic != 0)
      vnn_testing = new NN(*vnn, false, ::caffe::Phase::TEST);
    
    bestever_score = std::numeric_limits<double>::lowest();
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    std::copy(sensors.begin(), sensors.end(), last_state.begin());

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

  void update_critic(const caffe::Blob<double>& all_states, const caffe::Blob<double>& all_next_states,
    const caffe::Blob<double>& r_gamma_coef, const caffe::Blob<double>& taken_actions, 
    const caffe::Blob<double>& deter_next_actions) {
    
    if (trajectory.size() > 0) {
      caffe::Blob<double> v_target(trajectory.size(), 1, 1, 1);
      caffe::Blob<double> q_target(trajectory.size(), 1, 1, 1);

      auto iter = [&]() {
        auto all_nextQ = qnn->computeOutVFBlob(all_next_states, deter_next_actions);
        auto all_Q = qnn->computeOutVFBlob(all_states, taken_actions);
        //all_Q must be computed after all_nextQ to use learn_blob_no_full_forward

        decltype(vnn->computeOutVFBlob(all_states, empty_action)) all_nextV, all_V;
        if(batch_norm_critic != 0) {
          //learn BN
          all_V = vnn->computeOutVFBlob(all_states, empty_action);
          double* weights = new double[vnn->number_of_parameters(false)];
          vnn->copyWeightsTo(weights, false);
          vnn_testing->copyWeightsFrom(weights, false);
          delete[] weights;
          all_nextV = vnn_testing->computeOutVFBlob(all_next_states, empty_action);
          delete all_V;
          all_V = vnn_testing->computeOutVFBlob(all_states, empty_action);
        } else {
          all_nextV = vnn->computeOutVFBlob(all_next_states, empty_action);
          all_V = vnn->computeOutVFBlob(all_states, empty_action);
          //all_V must be computed after all_nextV to use learn_blob_no_full_forward
        }

#ifdef CAFFE_CPU_ONLY
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), v_target.cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), v_target.cpu_data(), all_V->cpu_data(), v_target.mutable_cpu_data());

        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextQ->cpu_data(), q_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), q_target.cpu_data(), q_target.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), q_target.cpu_data(), all_Q->cpu_data(), q_target.mutable_cpu_data());
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), v_target.cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), v_target.cpu_data(), all_V->cpu_data(), v_target.mutable_cpu_data());

        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextQ->cpu_data(), q_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), q_target.cpu_data(), q_target.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), q_target.cpu_data(), all_Q->cpu_data(), q_target.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), v_target.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), v_target.gpu_data(), v_target.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), v_target.gpu_data(), all_V->gpu_data(), v_target.mutable_gpu_data());

        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextQ->gpu_data(), q_target.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), q_target.gpu_data(), q_target.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), q_target.gpu_data(), all_Q->gpu_data(), q_target.mutable_gpu_data());
        break;
      }
#endif
        
//     Simple computation for lambda return
//    move v_target from GPU to CPU
        double* pdiff = v_target.mutable_cpu_diff();
        double* pdiffq = q_target.mutable_cpu_diff();
        const double* pvtarget = v_target.cpu_data();
        const double* pqtarget = q_target.cpu_data();
        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        double prev_deltaq = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it : trajectory) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              prev_deltaq = 0.;
              index_ep--;
          }
          pdiff[li] = pvtarget[li] + prev_delta;
          prev_delta = this->gamma * lambda * pdiff[li];

          pdiffq[li] = pqtarget[li] + prev_deltaq;
          prev_deltaq = this->gamma * lambdaQ * pdiffq[li];
          --li;
        }
        ASSERT(pdiff[trajectory.size() -1] == pvtarget[trajectory.size() -1], "pb lambda");
        ASSERT(pdiffq[trajectory.size() -1] == pqtarget[trajectory.size() -1], "pb lambda");
        
//         move diff to GPU
#ifdef CAFFE_CPU_ONLY
        caffe::caffe_add(trajectory.size(), v_target.cpu_diff(), all_V->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), q_target.cpu_diff(), all_Q->cpu_data(), q_target.mutable_cpu_data());
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_add(trajectory.size(), v_target.cpu_diff(), all_V->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), q_target.cpu_diff(), all_Q->cpu_data(), q_target.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_add(trajectory.size(), v_target.gpu_diff(), all_V->gpu_data(), v_target.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), q_target.gpu_diff(), all_Q->gpu_data(), q_target.mutable_gpu_data());
        break;
      }
#endif
        if (stoch_iter_critic == 1){
          vnn->learn_blob_no_full_forward(all_states, empty_action, v_target);
          qnn->learn_blob_no_full_forward(all_states, taken_actions, q_target);
        }
        else {
          vnn->learn_blob(all_states, empty_action, v_target, stoch_iter_critic);
          qnn->learn_blob(all_states, taken_actions, q_target, stoch_iter_critic);
        }

        delete all_V;
        delete all_nextV;

        delete all_Q;
        delete all_nextQ;
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

#ifdef PARALLEL_INTERACTION
    if (world.rank() == 0) {
      trajectory_end_points.clear();
      
      std::vector<std::deque<sample>> all_traj;
      std::vector<std::deque<int>> all_traj_ep;
      gather(world, trajectory, all_traj, 0);
      gather(world, trajectory_end_points, all_traj_ep, 0);

      ASSERT(all_traj.size() == all_traj_ep.size(), "pb");
      for (int i=0; i < all_traj.size() ; i++) {
        for (auto d2 : all_traj_ep[i])
          trajectory_end_points.push_back(trajectory.size() + d2);
        
        trajectory.insert(trajectory.end(), all_traj[i].begin(), all_traj[i].end());
      }
    } else {
      gather(world, trajectory, 0);
      gather(world, trajectory_end_points, 0);
    }

    //must be placed outside of if world...
    std::vector<std::vector<double>*> last_next_action(trajectory_end_points.size());
    
    if (world.rank() == 0) {
#else
    std::vector<std::vector<double>*> last_next_action(trajectory_end_points.size());
#endif
    
//
//    compute deter actions on last next state of trajectory
//
    for(int i=0; i < trajectory_end_points.size(); i++)
        last_next_action[i] = ann_testing->computeOut(trajectory[trajectory_end_points[i]-1].next_s);
//     
//    update norm on batch
//
    if (normalizer_type > 0) {
      for (int i=0;i<trajectory.size(); i++) {
        if(normalizer_type == 3)
          normalizer->update_batch_clip_before(trajectory[i].s);
        else
          normalizer->update_mean_var(trajectory[i].s);

        if(normalizer_action_type > 0)
            normalizer_action->update_mean_var(trajectory[i].a);
        else if (normalizer_action_type == 3)
            normalizer_action->update_batch_clip_before(trajectory[i].a);
      }
      
      for (int i=0;i<trajectory.size(); i++) {
        std::vector<double> normed_sensors(nb_sensors);
        std::vector<double> normed_next_s(nb_sensors);
        if (normalizer_type == 1) {
          normalizer->transform(normed_sensors, trajectory[i].s, false);
          normalizer->transform(normed_next_s, trajectory[i].next_s, false);
        } else if (normalizer_type == 2) {
          normalizer->transform_with_clip(normed_sensors, trajectory[i].s, false);
          normalizer->transform_with_clip(normed_next_s, trajectory[i].next_s, false);
        } else {
          normalizer->transform_with_double_clip(normed_sensors, trajectory[i].s, false);
          normalizer->transform_with_double_clip(normed_next_s, trajectory[i].next_s, false);
        }
        
        std::copy(normed_sensors.begin(), normed_sensors.end(), trajectory[i].s.begin());
        std::copy(normed_next_s.begin(), normed_next_s.end(), trajectory[i].next_s.begin());
      }
    }
#ifdef PARALLEL_INTERACTION
    }
    
//     synchronize normalizer
    if  (normalizer_type > 0) {
      bib::OnlineNormalizer on(this->nb_sensors);
      if (world.rank() == 0)
        on.copyFrom(*normalizer);
      
      broadcast(world, on, 0);
      
      if (world.rank() != 0)
        normalizer->copyFrom(on);
    }
    
    if (world.rank() == 0) {
#endif

    if(trajectory.size() > 0){
      vnn->increase_batchsize(trajectory.size());
      qnn->increase_batchsize(trajectory.size());
      if(batch_norm_critic != 0)
        vnn_testing->increase_batchsize(trajectory.size());
    }
    
    caffe::Blob<double> all_states(trajectory.size(), nb_sensors, 1, 1);
    caffe::Blob<double> all_next_states(trajectory.size(), nb_sensors, 1, 1);
    //store reward in data and gamma coef in diff
    caffe::Blob<double> r_gamma_coef(trajectory.size(), 1, 1, 1);
    caffe::Blob<double> taken_actions(trajectory.size(), this->nb_motors, 1, 1);
    caffe::Blob<double> deter_next_actions(trajectory.size(), this->nb_motors, 1, 1);
    
    double* pall_states = all_states.mutable_cpu_data();
    double* pdeter_next_actions = deter_next_actions.mutable_cpu_data();
    double* ptaken_actions = taken_actions.mutable_cpu_data();
    double* pdeter_actions = taken_actions.mutable_cpu_diff();
    double* pall_states_next = all_next_states.mutable_cpu_data();
    double* pr_all = r_gamma_coef.mutable_cpu_data();
    double* pgamma_coef = r_gamma_coef.mutable_cpu_diff();

    int li=0;
    int itedp = 0;
    for (auto it : trajectory) {
      std::copy(it.s.begin(), it.s.end(), pall_states + li * nb_sensors);
      std::copy(it.next_s.begin(), it.next_s.end(), pall_states_next + li * nb_sensors);
 
      if (normalizer_action_type == 0) {
        if( li + 1 != trajectory_end_points[itedp]) {
          if(learn_q_mu)
              std::copy(trajectory[li+1].pure_a.begin(), trajectory[li+1].pure_a.end(), pdeter_next_actions + (li) * this->nb_motors);
          else
              std::copy(trajectory[li+1].a.begin(), trajectory[li+1].a.end(), pdeter_next_actions + (li) * this->nb_motors);
        } else {
          std::copy(last_next_action[itedp]->begin(), last_next_action[itedp]->end(), pdeter_next_actions + (li) * this->nb_motors);
          itedp++;
        }
        std::copy(it.a.begin(), it.a.end(), ptaken_actions + li * this->nb_motors);
        std::copy(it.pure_a.begin(), it.pure_a.end(), pdeter_actions + li * this->nb_motors);
      } else {
        std::vector<double> normed_a(this->nb_motors);
        std::vector<double> normed_pure_a(this->nb_motors);

        std::vector<double> normed_a_next(this->nb_motors);
        if( li + 1 != trajectory_end_points[itedp]) {
          if(learn_q_mu)
            std::copy(trajectory[li+1].pure_a.begin(), trajectory[li+1].pure_a.end(), normed_a_next.begin());
          else
            std::copy(trajectory[li+1].a.begin(), trajectory[li+1].a.end(), normed_a_next.begin());
        } else {
          std::copy(last_next_action[itedp]->begin(), last_next_action[itedp]->end(), normed_a_next.begin());
          itedp++;
        }

        if (normalizer_action_type == 1) {
          normalizer_action->transform(normed_a, trajectory[li].a, false);
          normalizer_action->transform(normed_pure_a, trajectory[li].pure_a, false);
          
          normalizer_action->transform(normed_a_next, normed_a_next, false);
        } else if (normalizer_action_type == 2) {
          normalizer_action->transform_with_clip(normed_a, trajectory[li].a, false);
          normalizer_action->transform_with_clip(normed_pure_a, trajectory[li].pure_a, false);
          
          normalizer_action->transform_with_clip(normed_a_next, normed_a_next, false);
        } else {
          normalizer_action->transform_with_double_clip(normed_a, trajectory[li].a, false);
          normalizer_action->transform_with_double_clip(normed_pure_a, trajectory[li].pure_a, false);

          normalizer_action->transform_with_double_clip(normed_a_next, normed_a_next, false);
        }
        std::copy(normed_a_next.begin(), normed_a_next.end(), pdeter_next_actions + li * this->nb_motors);

        std::copy(normed_a.begin(), normed_a.end(), ptaken_actions + li * this->nb_motors);
        std::copy(normed_pure_a.begin(), normed_pure_a.end(), pdeter_actions + li * this->nb_motors);
      }
      pr_all[li]=it.r;
      pgamma_coef[li]= it.goal_reached ? 0.000f : this->gamma;
      li++;
    }
    ASSERT(itedp == trajectory_end_points.size(), "size pb");
//    bib::Logger::PRINT_ELEMENTS(pdeter_next_actions, deter_next_actions.count());

    for(int i=0; i < trajectory_end_points.size(); i++)
        delete last_next_action[i];

    update_critic(all_states, all_next_states, r_gamma_coef, taken_actions, deter_next_actions);
    

    if (trajectory.size() > 0) {
      const std::vector<double> disable_back_ac(this->nb_motors, 0.00f);
      caffe::Blob<double> deltas(trajectory.size(), 1, 1, 1);
      caffe::Blob<double> deltasQ(trajectory.size(), 1, 1, 1);

      caffe::caffe_copy(trajectory.size(), taken_actions.cpu_diff(), taken_actions.mutable_cpu_data());
      //taken actions now contains deter_actions
      auto all_nextQ = qnn->computeOutVFBlob(all_next_states, deter_next_actions);
      auto all_Q = qnn->computeOutVFBlob(all_states, taken_actions);

      decltype(vnn->computeOutVFBlob(all_next_states, empty_action)) all_nextV, all_mine;
      if(batch_norm_critic != 0)
      {
        double* weights = new double[vnn->number_of_parameters(false)];
        vnn->copyWeightsTo(weights, false);
        vnn_testing->copyWeightsFrom(weights, false);
        delete[] weights;
        all_nextV = vnn_testing->computeOutVFBlob(all_next_states, empty_action);
        all_mine = vnn_testing->computeOutVFBlob(all_states, empty_action);
      } else {
        all_nextV = vnn->computeOutVFBlob(all_next_states, empty_action);
        all_mine = vnn->computeOutVFBlob(all_states, empty_action);
      }

     
#ifdef CAFFE_CPU_ONLY
      caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), deltas.mutable_cpu_data());
      caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltas.cpu_data(), deltas.mutable_cpu_data());
      caffe::caffe_sub(trajectory.size(), deltas.cpu_data(), all_mine->cpu_data(), deltas.mutable_cpu_data());

      caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextQ->cpu_data(), deltasQ.mutable_cpu_data());
      caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltasQ.cpu_data(), deltasQ.mutable_cpu_data());
      caffe::caffe_sub(trajectory.size(), deltasQ.cpu_data(), all_Q->cpu_data(), deltasQ.mutable_cpu_data());
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), deltas.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltas.cpu_data(), deltas.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), deltas.cpu_data(), all_mine->cpu_data(), deltas.mutable_cpu_data());

        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextQ->cpu_data(), deltasQ.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltasQ.cpu_data(), deltasQ.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), deltasQ.cpu_data(), all_Q->cpu_data(), deltasQ.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), deltas.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), deltas.gpu_data(), deltas.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), deltas.gpu_data(), all_mine->gpu_data(), deltas.mutable_gpu_data());

        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextQ->gpu_data(), deltasQ.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), deltasQ.gpu_data(), deltasQ.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), deltasQ.gpu_data(), all_Q->gpu_data(), deltasQ.mutable_gpu_data());
        break;
      }
#endif
 
      if(gae){
        //        Simple computation for lambda return
        //        move deltas from GPU to CPU
        double * diff = deltas.mutable_cpu_diff();
        const double* pdeltas = deltas.cpu_data();

        double * diffQ = deltasQ.mutable_cpu_diff();
        const double* pdeltasQ = deltasQ.cpu_data();

        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        double prev_deltaQ = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it : trajectory) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              prev_deltaQ = 0.;
              index_ep--;
          }
          diff[li] = pdeltas[li] + prev_delta;
          prev_delta = this->gamma * lambda * diff[li];

          diffQ[li] = pdeltasQ[li] + prev_deltaQ;
          prev_deltaQ = this->gamma * lambdaQ * diffQ[li];
          --li;
        }
        ASSERT(diff[trajectory.size() -1] == pdeltas[trajectory.size() -1], "pb lambda");
        ASSERT(diffQ[trajectory.size() -1] == pdeltasQ[trajectory.size() -1], "pb lambda");

        caffe::caffe_copy(trajectory.size(), deltas.cpu_diff(), deltas.mutable_cpu_data());
        caffe::caffe_copy(trajectory.size(), deltasQ.cpu_diff(), deltasQ.mutable_cpu_data());
      }
      
      uint n=0;
      posdelta_mean=0.f;
      //store target in data, and disable in diff
      caffe::Blob<double> target_cac(trajectory.size(), this->nb_motors, 1, 1);
      caffe::caffe_set(target_cac.count(), static_cast<double>(1.f), target_cac.mutable_cpu_diff());
      caffe::Blob<double> deltas_blob(trajectory.size(), this->nb_motors, 1, 1);
      caffe::caffe_set(deltas_blob.count(), static_cast<double>(1.f), deltas_blob.mutable_cpu_data());

      double* pdisable_back_cac = target_cac.mutable_cpu_diff();
      double* pdeltas_blob = deltas_blob.mutable_cpu_data();
      double* ptarget_cac = target_cac.mutable_cpu_data();
      const double* pdeltas = deltas.cpu_data();
      const double* pdeltasQ = deltasQ.cpu_data();
      li=0;
      //cacla cost
      // (mu(s) - a) A(s,a) H(A(s,a))
      // clip(mu(s) - a, mu_old(s) - c, mu_old(s) + c) A(s,a) H(A(s,a)) clip only direction not target
      // mu(s) - clip(a, mu_old(s) - c, mu_old(s) + c) A(s,a) H(A(s,a))

      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        for(int j=0; j < this->nb_motors; j++)
            ptarget_cac[li * this->nb_motors + j] = std::min( std::min(it->pure_a[j] + beta_target, (double) 1.f), std::max(it->a[j], std::max(it->pure_a[j] - beta_target, (double) -1.f) ) );
//        if(pdeltas[li] > 0.) {
        if(pdeltasQ[li] > 0.) {
          posdelta_mean += pdeltas[li];
          n++;
        } else {
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), pdisable_back_cac + li * this->nb_motors);
        }
        if(!disable_cac)
            std::fill(pdeltas_blob + li * this->nb_motors, pdeltas_blob + (li+1) * this->nb_motors, pdeltasQ[li]);
        li++;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      posdelta_mean = posdelta_mean / ((float) trajectory.size());
      int size_cost_cacla=trajectory.size()*this->nb_motors;
      
      if(n > 0) {
        for(uint sia = 0; sia < stoch_iter_actor; sia++){
          ann->increase_batchsize(trajectory.size());
          //learn BN
          auto ac_out = ann->computeOutBlob(all_states);
          if(batch_norm_actor != 0) {
            //re-compute ac_out with BN as testing
            double* weights = new double[ann->number_of_parameters(false)];
            ann->copyWeightsTo(weights, false);
            ann_testing_blob->copyWeightsFrom(weights, false);
            delete[] weights;
            delete ac_out;
            ann_testing_blob->increase_batchsize(trajectory.size());
            ac_out = ann_testing_blob->computeOutBlob(all_states);
          }
          ann->ZeroGradParameters();
          
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          
          caffe::Blob<double> diff_cac(trajectory.size(), this->nb_motors, 1, 1);
          double * ac_diff = nullptr;
#ifdef CAFFE_CPU_ONLY
          ac_diff = actor_actions_blob->mutable_cpu_diff();
          caffe::caffe_sub(size_cost_cacla, target_cac.cpu_data(), ac_out->cpu_data(), diff_cac.mutable_cpu_data());
          caffe::caffe_mul(size_cost_cacla, diff_cac.cpu_data(), deltas_blob.cpu_data(), diff_cac.mutable_cpu_data());
          caffe::caffe_mul(size_cost_cacla, target_cac.cpu_diff(), diff_cac.cpu_data(), ac_diff);
          caffe::caffe_scal(size_cost_cacla, (double) -1.f, ac_diff);
#else
          switch (caffe::Caffe::mode()) {
          case caffe::Caffe::CPU:
            ac_diff = actor_actions_blob->mutable_cpu_diff();
            caffe::caffe_sub(size_cost_cacla, target_cac.cpu_data(), ac_out->cpu_data(), diff_cac.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_cac.cpu_data(), deltas_blob.cpu_data(), diff_cac.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, target_cac.cpu_diff(), diff_cac.cpu_data(), ac_diff);
            caffe::caffe_scal(size_cost_cacla, (double) -1.f, ac_diff);
            break;
          case caffe::Caffe::GPU:
            ac_diff = actor_actions_blob->mutable_gpu_diff();
            caffe::caffe_gpu_sub(size_cost_cacla, target_cac.gpu_data(), ac_out->gpu_data(), diff_cac.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, diff_cac.gpu_data(), deltas_blob.gpu_data(), diff_cac.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, target_cac.gpu_diff(), diff_cac.gpu_data(), ac_diff);
            caffe::caffe_gpu_scal(size_cost_cacla, (double) -1.f, ac_diff);
            break;
          }
#endif

          ann->actor_backward();
          ann->updateFisher(n);
          ann->regularize();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
          delete ac_out;
        }
      } else if(batch_norm_actor != 0){
        //learn BN even if every action were bad
        ann->increase_batchsize(trajectory.size());
        delete ann->computeOutBlob(all_states);
      }
      
      delete all_nextV;
      delete all_mine;

      delete all_nextQ;
      delete all_Q;
      
      if(batch_norm_actor != 0)
        ann_testing->increase_batchsize(1);
    }
#ifdef PARALLEL_INTERACTION
    }
    std::vector<double> weights(ann->number_of_parameters(false), 0.f);
    if (world.rank() == 0)
        ann->copyWeightsTo(weights.data(), false);

    broadcast(world, weights, 0);
    if (world.rank() != 0)
        ann->copyWeightsFrom(weights.data(), false);
#endif
    nb_sample_update = trajectory.size();
    trajectory.clear();
    trajectory_end_points.clear();
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
#ifdef PARALLEL_INTERACTION
      if(world.rank() == 0 ) {
#endif
        ann->save(path+".actor");
        vnn->save(path+".critic");
        if (normalizer_type > 0)
            bib::XMLEngine::save<>(*normalizer, "normalizer", path+".normalizer.data");
#ifdef PARALLEL_INTERACTION
      }
#endif
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
    if (normalizer_type > 0){
        delete normalizer;
        normalizer = bib::XMLEngine::load<bib::OnlineNormalizer>("normalizer", path+".normalizer.data");
    }
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

#ifdef PARALLEL_INTERACTION
  int getMPIrank() {
    return world.rank();
  }
#endif

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
    nb_sample_update << " " << std::setprecision(3) << ratio_valid_advantage << " " << conserved_l2dist << " " << 
    std::setprecision(3) << vnn->weight_l1_norm() << " " << ann->weight_l1_norm(true) << " " << posdelta_mean;
  }
  
 private:
  uint nb_sensors;
  uint episode = 1;
  uint step = 0;

  double noise, noise2, noise3;
  uint gaussian_policy;
  bool gae, disable_cac, learn_q_mu;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, lambdaQ, beta_target;
  double conserved_l2dist= 0.f;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::deque<sample> trajectory;
  std::deque<int> trajectory_end_points;

  NN* ann;
  NN* vnn;
  NN* ann_testing;
  NN* ann_testing_blob;
  NN* vnn_testing;
  NN* qnn;

  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  caffe::Blob<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  float ratio_valid_advantage=0;
  int nb_sample_update = 0;
  double posdelta_mean = 0;
  
  uint normalizer_type;
  uint normalizer_action_type;
  bib::OnlineNormalizer* normalizer = nullptr;
  bib::OnlineNormalizer* normalizer_action = nullptr;

#ifdef PARALLEL_INTERACTION
  boost::mpi::communicator world;
#endif
  
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


