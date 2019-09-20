#ifndef PENNFAC_HPP
#define PENNFAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <caffe/util/math_functions.hpp>

//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>
//#include "lapacke.h"

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
    delete vnn;
    delete ann;
    
    delete ann_testing;
    if(batch_norm_actor != 0)
      delete ann_testing_blob;
    if(batch_norm_critic != 0)
      delete vnn_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
    
    if(normalizer_type > 0)
      delete normalizer;
    
    if(oun == nullptr)
      delete oun;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors_,
                                  bool learning, bool goal_reached, bool) override {

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
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<uint>("agent.gaussian_policy");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");
    stoch_iter_actor        = pt->get<uint>("agent.stoch_iter_actor");
    stoch_iter_critic       = pt->get<uint>("agent.stoch_iter_critic");
    stoch_iter_aux          = pt->get<uint>("agent.stoch_iter_aux");
    batch_norm_actor        = pt->get<uint>("agent.batch_norm_actor");
    batch_norm_critic       = pt->get<uint>("agent.batch_norm_critic");
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_aux                 = pt->get<double>("agent.alpha_aux");
    alpha_v                 = pt->get<double>("agent.alpha_v");
    lambda                  = pt->get<double>("agent.lambda");
    momentum                = pt->get<uint>("agent.momentum");
    beta_target             = pt->get<double>("agent.beta_target");
    ignore_poss_ac          = pt->get<bool>("agent.ignore_poss_ac");
    conserve_beta           = pt->get<bool>("agent.conserve_beta");
    disable_trust_region = pt->get<bool>("agent.disable_trust_region");
    disable_cac                 = pt->get<bool>("agent.disable_cac");
    correlation_importance = pt->get<double>("agent.correlation_importance");
    correlation_history = pt->get<uint>("agent.correlation_history");
    update_aux_before   = pt->get<bool>("agent.update_aux_before");
    gae                     = false;
    update_each_episode = 1;
    normalizer_type = 0;
    
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
    
    if(lambda >= 0.)
      gae = pt->get<bool>("agent.gae");
    
    if(lambda >=0. && batch_norm_critic != 0){
      LOG_DEBUG("to be done!");
      exit(1);
    }
    
    try {
      normalizer_type     = pt->get<uint>("agent.normalizer_type");
    } catch(boost::exception const& ) {
    }
    if(normalizer_type > 0)
      normalizer = new bib::OnlineNormalizer(this->nb_sensors);

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

    aux = new NN(this->nb_motors * (1 + correlation_history), {}, this->nb_motors, alpha_aux, 1, hidden_layer_type, 0, 0, true, momentum);

#ifdef PARALLEL_INTERACTION
    {
      std::vector<double> weights(ann->number_of_parameters(false), 0.f);
      if (world.rank() == 0)
          ann->copyWeightsTo(weights.data(), false);

      broadcast(world, weights, 0);
      if (world.rank() != 0)
          ann->copyWeightsFrom(weights.data(), false);
    }
#endif
    
    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm_critic, false, momentum);
    
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
    const caffe::Blob<double>& r_gamma_coef) {
    
    if (trajectory.size() > 0) {
      caffe::Blob<double> v_target(trajectory.size(), 1, 1, 1);

      //remove trace of old policy
      auto iter = [&]() {
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
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), v_target.cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), v_target.cpu_data(), all_V->cpu_data(), v_target.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), v_target.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), v_target.gpu_data(), v_target.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), v_target.gpu_data(), all_V->gpu_data(), v_target.mutable_gpu_data());
        break;
      }
#endif
        
//     Simple computation for lambda return
//    move v_target from GPU to CPU
        double* pdiff = v_target.mutable_cpu_diff();
        const double* pvtarget = v_target.cpu_data();
        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it : trajectory) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              index_ep--;
          }
          pdiff[li] = pvtarget[li] + prev_delta;
          prev_delta = this->gamma * lambda * pdiff[li];
          --li;
        }
        ASSERT(pdiff[trajectory.size() -1] == pvtarget[trajectory.size() -1], "pb lambda");
        
//         move diff to GPU
#ifdef CAFFE_CPU_ONLY
        caffe::caffe_add(trajectory.size(), v_target.cpu_diff(), all_V->cpu_data(), v_target.mutable_cpu_data());
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_add(trajectory.size(), v_target.cpu_diff(), all_V->cpu_data(), v_target.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_add(trajectory.size(), v_target.gpu_diff(), all_V->gpu_data(), v_target.mutable_gpu_data());
        break;
      }
#endif
        if (stoch_iter_critic == 1)
          vnn->learn_blob_no_full_forward(all_states, empty_action, v_target);
        else
          vnn->learn_blob(all_states, empty_action, v_target, stoch_iter_critic);

        delete all_V;
        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();
    }
  }
  
  void update_aux() {
    caffe::Blob<double> all_deter_action_out(trajectory.size() - (correlation_history * trajectory_end_points.size()), this->nb_motors, 1, 1);
    caffe::Blob<double> all_deter_action_in(trajectory.size() - (correlation_history * trajectory_end_points.size()), this->nb_motors * (1 + correlation_history), 1, 1);
    int li=0;
    int li2=0;
    double* pall_deter_action_out = all_deter_action_out.mutable_cpu_data();
    double* pall_deter_action_in = all_deter_action_in.mutable_cpu_data();
    for (int i=0; i < trajectory_end_points.size(); i++){
        int beg = 0;
        int end = trajectory_end_points[i];
        if (i - 1 >= 0)
          beg += trajectory_end_points[i-1];
        
        for (int j=beg; j < end; j++) {
          if (j+correlation_history < end){
            std::copy(trajectory[j+correlation_history].pure_a.begin(), trajectory[j+correlation_history].pure_a.end(), pall_deter_action_out + li * this->nb_motors);
            li++;
          }
          
          if (j + correlation_history < end) {
            for(int k=0;k< 1 + correlation_history ; k++) {
              std::copy(trajectory[j+k].pure_a.begin(), trajectory[j+k].pure_a.end(), pall_deter_action_in + li2 * this->nb_motors);
              li2++;
            }
          }
        }
    }
    ASSERT(li == (int) ((double)all_deter_action_out.count() / this->nb_motors), "pb size " << li << " " << (int) ((double)all_deter_action_out.count() / this->nb_motors));
    ASSERT(li2 == (int) ((double)all_deter_action_in.count() / this->nb_motors), "pb size " << li2 << " " << (int) ((double)all_deter_action_in.count() / this->nb_motors));
    
//     bib::Logger::PRINT_ELEMENTS(trajectory_end_points);
//     li=0;
//     for (auto it : trajectory) {
//       bib::Logger::PRINT_ELEMENTS(it.pure_a, std::to_string(li).c_str());
//       li++;
//     }
//     
//     bib::Logger::PRINT_ELEMENTS(pall_deter_action_out, all_deter_action_out.count() , " out ");
//     bib::Logger::PRINT_ELEMENTS(pall_deter_action_in,  all_deter_action_in.count(), " in ");

    aux->increase_batchsize(trajectory.size() - (correlation_history * trajectory_end_points.size()));
    for (int i=0; i  < stoch_iter_aux; i++) {
      aux->learn_blob(all_deter_action_in, empty_action, all_deter_action_out, 1);

      //enforce diag to be null (or solution is obvious)
      double* weights = aux->getNN()->learnable_params()[0]->mutable_cpu_data();
      for (int i=0; i < this->nb_motors ;i++)
          weights[i+ i * this->nb_motors] = 0.0f;
      
//       if( i% 100 == 0)
//         LOG_DEBUG(i << " " << std::setprecision(12) << aux->error() );
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
    
    if (world.rank() == 0) {
#endif
    
//     
//    update norm on batch
//
    if (normalizer_type > 0) {
      for (int i=0;i<trajectory.size(); i++) {
        if(normalizer_type == 3)
          normalizer->update_batch_clip_before(trajectory[i].s);
        else
          normalizer->update_mean_var(trajectory[i].s);
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
      if(batch_norm_critic != 0)
        vnn_testing->increase_batchsize(trajectory.size());
    }
    
    caffe::Blob<double> all_states(trajectory.size(), nb_sensors, 1, 1);
    caffe::Blob<double> all_next_states(trajectory.size(), nb_sensors, 1, 1);
    //store reward in data and gamma coef in diff
    caffe::Blob<double> r_gamma_coef(trajectory.size(), 1, 1, 1);
    
    double* pall_states = all_states.mutable_cpu_data();
    double* pall_states_next = all_next_states.mutable_cpu_data();
    double* pr_all = r_gamma_coef.mutable_cpu_data();
    double* pgamma_coef = r_gamma_coef.mutable_cpu_diff();

    int li=0;
    for (auto it : trajectory) {
      std::copy(it.s.begin(), it.s.end(), pall_states + li * nb_sensors);
      std::copy(it.next_s.begin(), it.next_s.end(), pall_states_next + li * nb_sensors);
      pr_all[li]=it.r;
      pgamma_coef[li]= it.goal_reached ? 0.000f : this->gamma;
      li++;
    }
    
    if (update_aux_before)
      update_aux();
    
    update_critic(all_states, all_next_states, r_gamma_coef);

    if (trajectory.size() > 0) {
      const std::vector<double> disable_back_ac(this->nb_motors, 0.00f);
      caffe::Blob<double> deltas(trajectory.size(), 1, 1, 1);

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
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), deltas.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltas.cpu_data(), deltas.mutable_cpu_data());
        caffe::caffe_sub(trajectory.size(), deltas.cpu_data(), all_mine->cpu_data(), deltas.mutable_cpu_data());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), deltas.mutable_gpu_data());
        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), deltas.gpu_data(), deltas.mutable_gpu_data());
        caffe::caffe_gpu_sub(trajectory.size(), deltas.gpu_data(), all_mine->gpu_data(), deltas.mutable_gpu_data());
        break;
      }
#endif
 
      if(gae){
        //        Simple computation for lambda return
        //        move deltas from GPU to CPU
        double * diff = deltas.mutable_cpu_diff();
        const double* pdeltas = deltas.cpu_data();
        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it : trajectory) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              index_ep--;
          }
          diff[li] = pdeltas[li] + prev_delta;
          prev_delta = this->gamma * lambda * diff[li];
          --li;
        }
        ASSERT(diff[trajectory.size() -1] == pdeltas[trajectory.size() -1], "pb lambda");

        caffe::caffe_copy(trajectory.size(), deltas.cpu_diff(), deltas.mutable_cpu_data());
      }
      
      uint n=0;
      posdelta_mean=0.f;
      //store target in data, and disable in diff
      caffe::Blob<double> target_cac(trajectory.size(), this->nb_motors, 1, 1);
      caffe::Blob<double> target_treg(trajectory.size(), this->nb_motors, 1, 1);
      caffe::caffe_set(target_cac.count(), static_cast<double>(1.f), target_cac.mutable_cpu_diff());
      caffe::caffe_set(target_treg.count(), static_cast<double>(1.f), target_treg.mutable_cpu_diff());
      caffe::Blob<double> deltas_blob(trajectory.size(), this->nb_motors, 1, 1);
      caffe::caffe_set(deltas_blob.count(), static_cast<double>(1.f), deltas_blob.mutable_cpu_data());

      double* pdisable_back_cac = target_cac.mutable_cpu_diff();
      double* pdisable_back_treg = target_treg.mutable_cpu_diff();
      double* pdeltas_blob = deltas_blob.mutable_cpu_data();
      double* ptarget_cac = target_cac.mutable_cpu_data();
      double* ptarget_treg = target_treg.mutable_cpu_data();
      const double* pdeltas = deltas.cpu_data();
      li=0;
      //cacla cost
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        std::copy(it->a.begin(), it->a.end(), ptarget_cac + li * this->nb_motors);
        if(pdeltas[li] > 0.) {
          posdelta_mean += pdeltas[li];
          n++;
        } else {
          std::copy(disable_back_ac.begin(), disable_back_ac.end(), pdisable_back_cac + li * this->nb_motors);
        }
        if(!disable_cac)
            std::fill(pdeltas_blob + li * this->nb_motors, pdeltas_blob + (li+1) * this->nb_motors, pdeltas[li]);
        li++;
      }
      //penalty cost
      li=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        std::copy(it->pure_a.begin(), it->pure_a.end(), ptarget_treg + li * this->nb_motors);
        if(ignore_poss_ac && pdeltas[li] > 0.) {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), pdisable_back_treg + li * this->nb_motors);
        }
        li++;
      }

      ratio_valid_advantage = ((float)n) / ((float) trajectory.size());
      posdelta_mean = posdelta_mean / ((float) trajectory.size());
      int size_cost_cacla=trajectory.size()*this->nb_motors;
      
      double beta=0.0001f;
      mean_beta=0.f;
      if(conserve_beta)
        beta=conserved_beta;
      mean_beta += beta;
      
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
          
          number_effective_actor_update = sia;
          if(disable_trust_region)
              beta=0.f;
          else if (sia > 0) {
            //compute deter distance(pi, pi_old)
            caffe::Blob<double> diff_treg(trajectory.size(), this->nb_motors, 1, 1);
            double l2distance = 0.f;
#ifdef CAFFE_CPU_ONLY
            caffe::caffe_sub(size_cost_cacla, target_treg.cpu_data(), ac_out->cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_treg.cpu_data(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            l2distance = caffe::caffe_cpu_asum(size_cost_cacla, diff_treg.cpu_data());
#else
          switch (caffe::Caffe::mode()) {
          case caffe::Caffe::CPU:
            caffe::caffe_sub(size_cost_cacla, target_treg.cpu_data(), ac_out->cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_treg.cpu_data(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            l2distance = caffe::caffe_cpu_asum(size_cost_cacla, diff_treg.cpu_data());
            break;
          case caffe::Caffe::GPU:
            caffe::caffe_gpu_sub(size_cost_cacla, target_treg.gpu_data(), ac_out->gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, diff_treg.gpu_data(), diff_treg.gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_asum(size_cost_cacla, diff_treg.gpu_data(), &l2distance);
            break;
          }
#endif
            l2distance = std::sqrt(l2distance/((double) trajectory.size()*this->nb_motors));

            if (l2distance < beta_target/1.5)
                beta = beta/2.;
            else if (l2distance > beta_target*1.5)
                beta = beta*2.;

            beta=std::max(std::min((double)20.f, beta), (double) 0.01f);
            mean_beta += beta;
            conserved_l2dist = l2distance;
            //LOG_DEBUG(std::setprecision(7) << l2distance << " " << beta << " " << beta_target << " " << sia);
          }
          
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          
          //correlation cost
          li = 0;
          int li2 = 0;
          const double* pac_out = ac_out->cpu_data();
          caffe::Blob<double> aux_in(trajectory.size() - (correlation_history * trajectory_end_points.size()), this->nb_motors * (1 + correlation_history), 1, 1);
          caffe::Blob<double> aux_out(trajectory.size() - (correlation_history * trajectory_end_points.size()), this->nb_motors, 1, 1);
          double* paux_in = aux_in.mutable_cpu_data();
          double* paux_out = aux_out.mutable_cpu_data();
          for (int i=0; i < trajectory_end_points.size(); i++) {
            int beg = 0;
            int end = trajectory_end_points[i];
            if (i - 1 >= 0)
              beg += trajectory_end_points[i-1];
            
            for (int j=beg; j < end; j++) {
              if (j+correlation_history < end) {
                std::copy(pac_out + (j+correlation_history) *  this->nb_motors, pac_out + (j+correlation_history+1) *  this->nb_motors, paux_out + li * this->nb_motors);
                li++;
              }
              
              if (j + correlation_history < end) {
                for(int k=0;k< 1 + correlation_history ; k++) {
                  std::copy(pac_out + (j+k) * this->nb_motors, pac_out + (j+k+1) * this->nb_motors, paux_in + li2 * this->nb_motors);
                  li2++;
                }
              }
            }
          }
//          bib::Logger::PRINT_ELEMENTS(pac_out, ac_out->count(), "in");
//          bib::Logger::PRINT_ELEMENTS(paux_in, aux_in.count(), "in adapt");
//          bib::Logger::PRINT_ELEMENTS(trajectory_end_points, "tep ");
          ASSERT(li == (int) ((double)aux_out.count() / this->nb_motors), "pb size " << li << " " << (int) ((double)aux_out.count() / this->nb_motors));
          ASSERT(li2 == (int) ((double)aux_in.count() / this->nb_motors), "pb size " << li2 << " " << (int) ((double)aux_in.count() / this->nb_motors));
          aux->increase_batchsize(trajectory.size() - (correlation_history * trajectory_end_points.size()));
          auto aux_out2 = aux->computeOutBlob(aux_in);
          const double* paux_out2 = aux_out2->cpu_data();
          const auto aux_output_blob = aux->getNN()->blob_by_name(MLP::actions_blob_name);
          double* aux_diff = aux_output_blob->mutable_cpu_diff();
          for (int i=0;i < aux_output_blob->count(); i++)
            aux_diff[i] = - (paux_out2[i] - paux_out[i]);
//          bib::Logger::PRINT_ELEMENTS(aux_diff, aux_output_blob->count());
          aux->actor_backward();
          const auto aux_input_blob = aux->getNN()->blob_by_name(MLP::states_blob_name);
          
//          LOG_DEBUG(aux_input_blob->count() << " " << actor_actions_blob->count());
          
          caffe::Blob<double> diff_corr(trajectory.size(), this->nb_motors, 1, 1);
          caffe::caffe_set(diff_corr.count(), static_cast<double>(0.0f), diff_corr.mutable_cpu_data());
          li = 0;
          double* pdiff_corr = diff_corr.mutable_cpu_data();
          const double* pdiff_aux_in = aux_input_blob->cpu_diff();
          
          int index_ep = 0;
          int removed_i = 0;
          std::vector<int> counter(trajectory.size(), 0);
          for(int i=0; i < aux_input_blob->count() / this->nb_motors; i++){
              int adapted_i = i - removed_i;
              int sample_index = (adapted_i / (1+correlation_history)) + (adapted_i %(1+correlation_history));
 
              if (index_ep - 1 >= 0)
                sample_index += trajectory_end_points[index_ep-1];
              
              if (sample_index + 1 == trajectory_end_points[index_ep]){
                index_ep ++;
                removed_i = i + 1;
              }
             
//              LOG_DEBUG(i << " " << sample_index);
              for (int k=0; k < this->nb_motors; k++)
                pdiff_corr[ sample_index*this->nb_motors + k ] += pdiff_aux_in[ i*this->nb_motors+k ];
              counter[sample_index] ++;
          }
          ASSERT(index_ep == trajectory_end_points.size(), index_ep << " " <<  trajectory_end_points.size());
          for(int i=0; i < trajectory.size(); i++){
            for (int k=0; k < this->nb_motors; k++) {
              pdiff_corr[i*this->nb_motors + k] *= correlation_importance * (1.f / (double) counter[i]);
//               if(std::fabs(pdiff_corr[i*this->nb_motors + k] ) >= 10)
//                 LOG_DEBUG("catch");
            }
          }
          delete aux_out2;
          
          caffe::Blob<double> diff_cac(trajectory.size(), this->nb_motors, 1, 1);
          caffe::Blob<double> diff_treg(trajectory.size(), this->nb_motors, 1, 1);
          double * ac_diff = nullptr;
#ifdef CAFFE_CPU_ONLY
          ac_diff = actor_actions_blob->mutable_cpu_diff();
          caffe::caffe_sub(size_cost_cacla, target_cac.cpu_data(), ac_out->cpu_data(), diff_cac.mutable_cpu_data());
          caffe::caffe_mul(size_cost_cacla, diff_cac.cpu_data(), deltas_blob.cpu_data(), diff_cac.mutable_cpu_data());
          caffe::caffe_mul(size_cost_cacla, target_cac.cpu_diff(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
          
          caffe::caffe_sub(size_cost_cacla, target_treg.cpu_data(), ac_out->cpu_data(), diff_treg.mutable_cpu_data());
          caffe::caffe_scal(size_cost_cacla, beta, diff_treg.mutable_cpu_data());
          caffe::caffe_mul(size_cost_cacla, target_treg.cpu_diff(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
          
          caffe::caffe_add(size_cost_cacla, diff_corr.cpu_data(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
          caffe::caffe_add(size_cost_cacla, diff_cac.cpu_data(), diff_treg.cpu_data(), ac_diff);
          caffe::caffe_scal(size_cost_cacla, (double) -1.f, ac_diff);
#else
          switch (caffe::Caffe::mode()) {
          case caffe::Caffe::CPU:
            ac_diff = actor_actions_blob->mutable_cpu_diff();
            caffe::caffe_sub(size_cost_cacla, target_cac.cpu_data(), ac_out->cpu_data(), diff_cac.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_cac.cpu_data(), deltas_blob.cpu_data(), diff_cac.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, target_cac.cpu_diff(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
            
            caffe::caffe_sub(size_cost_cacla, target_treg.cpu_data(), ac_out->cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_scal(size_cost_cacla, beta, diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, target_treg.cpu_diff(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            
            caffe::caffe_add(size_cost_cacla, diff_corr.cpu_data(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
            caffe::caffe_add(size_cost_cacla, diff_cac.cpu_data(), diff_treg.cpu_data(), ac_diff);
            caffe::caffe_scal(size_cost_cacla, (double) -1.f, ac_diff);
            break;
          case caffe::Caffe::GPU:
            ac_diff = actor_actions_blob->mutable_gpu_diff();
            caffe::caffe_gpu_sub(size_cost_cacla, target_cac.gpu_data(), ac_out->gpu_data(), diff_cac.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, diff_cac.gpu_data(), deltas_blob.gpu_data(), diff_cac.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, target_cac.gpu_diff(), diff_cac.gpu_data(), diff_cac.mutable_gpu_data());
            
            caffe::caffe_gpu_sub(size_cost_cacla, target_treg.gpu_data(), ac_out->gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_scal(size_cost_cacla, beta, diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, target_treg.gpu_diff(), diff_treg.gpu_data(), diff_treg.mutable_gpu_data());
            
            caffe::caffe_gpu_add(size_cost_cacla, diff_corr.cpu_data(), diff_cac.gpu_data(), diff_cac.mutable_gpu_data());
            caffe::caffe_gpu_add(size_cost_cacla, diff_cac.gpu_data(), diff_treg.gpu_data(), ac_diff);
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
      
      conserved_beta = beta;
      mean_beta /= (double) number_effective_actor_update;

      delete all_nextV;
      delete all_mine;
      
      if(batch_norm_actor != 0)
        ann_testing->increase_batchsize(1);
    }
    
    if (!update_aux_before)
      update_aux();
    
#ifdef PARALLEL_INTERACTION
    }
    {
      std::vector<double> weights(ann->number_of_parameters(false), 0.f);
      if (world.rank() == 0)
          ann->copyWeightsTo(weights.data(), false);

      broadcast(world, weights, 0);
      if (world.rank() != 0)
          ann->copyWeightsFrom(weights.data(), false);
    }
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
      aux->save(path+".aux");
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
    aux->load(path+".aux");
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
  bool gae, ignore_poss_ac, conserve_beta, disable_trust_region, disable_cac, update_aux_before;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic, stoch_iter_aux;
  uint batch_norm_actor, batch_norm_critic, actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, beta_target;
  double conserved_beta= 0.0001f;
  double mean_beta= 0.f;
  double conserved_l2dist= 0.f;
  int number_effective_actor_update = 0;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a, alpha_aux;

  std::deque<sample> trajectory;
  std::deque<int> trajectory_end_points;

  NN* ann;
  NN* aux;
  NN* vnn;
  NN* ann_testing;
  NN* ann_testing_blob;
  NN* vnn_testing;

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
  bib::OnlineNormalizer* normalizer = nullptr;
  
  double correlation_importance;
  uint correlation_history;

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

