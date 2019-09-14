#ifndef HPENFACAG_HPP_INCLUDED
#define HPENFACAG_HPP_INCLUDED

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
  std::vector<double> goal_achieved;
  std::vector<double> goal_achieved_unnormed;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  std::vector<double> next_goal_achieved_unnormed;
  double r;
  bool goal_reached;
  double prob;
  bool artificial;
  bool interest;
  
  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(goal_achieved);
    ar& BOOST_SERIALIZATION_NVP(goal_achieved_unnormed);
    ar& BOOST_SERIALIZATION_NVP(pure_a);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(next_goal_achieved_unnormed);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
    ar& BOOST_SERIALIZATION_NVP(prob);
    ar& BOOST_SERIALIZATION_NVP(artificial);
    ar& BOOST_SERIALIZATION_NVP(interest);
  }
} sample;
#endif

template<typename NN = MLP>
class OfflineCaclaAg : public arch::AACAgent<NN, arch::AgentGPUProgOptions> {
 public:
  typedef NN PolicyImpl;

  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors, uint _goal_size, uint _goal_start)
      : arch::AACAgent<NN, arch::AgentGPUProgOptions>(_nb_motors, _nb_sensors-_goal_size), nb_sensors(_nb_sensors-_goal_size), empty_action(), 
      last_state(_nb_sensors-_goal_size, 0.f), last_goal_achieved(_goal_size, 0.f), goal_size(_goal_size), goal_start(_goal_start-_goal_size),
      normalizer(nb_sensors) {

        LOG_DEBUG("goal size " << goal_size);
        LOG_DEBUG("goal start " << goal_start);
  }

  virtual ~OfflineCaclaAg() {
    if(vnn != nullptr)
      delete vnn;
    delete ann;
    delete ann_noblob;

    delete hidden_unit_v;
    delete hidden_unit_a;
    
    if(oun == nullptr)
      delete oun;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  const std::vector<double>& goal_achieved, bool learning, bool as, bool) override {
    std::vector<double> normed_sensors(nb_sensors);
    normalizer.transform_with_double_clip(normed_sensors, sensors, false);
    
    // protect batch norm from testing data and poor data
    vector<double>* next_action = ann_noblob->computeOut(normed_sensors);
    if (last_action.get() != nullptr && learning) {
      double prob = bib::Proba<double>::truncatedGaussianDensity(*last_action, *last_pure_action, noise);
      bool gr = reward >= -0.0000001;
      trajectory.push_back( {last_state, last_goal_achieved, last_goal_achieved, *last_pure_action, *last_action, sensors, goal_achieved, reward, gr, prob, false, true});
      if (gr)
        trajectory_end_points.push_back(trajectory.size());
      
//      auto sa = trajectory.back();
//      double nr = dense_reward(sa.goal_achieved_unnormed, sa.s,
//                                                    sa.next_goal_achieved_unnormed, sa.next_s,
//                                                    sa.s, sa.next_s);
//      LOG_DEBUG(reward << " " << nr);
    }

    last_pure_action.reset(new std::vector<double>(*next_action));
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
      } else if(gaussian_policy == 5) {
        if (bib::Utils::rand01() < noise2){
          for (uint i = 0; i < next_action->size(); i++)
            next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
        } else {
          vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
          delete next_action;
          next_action = randomized_action;
        }
      } else if(bib::Utils::rand01() < noise) { //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);

    std::copy(sensors.begin(), sensors.end(), last_state.begin());
    std::copy(goal_achieved.begin(), goal_achieved.end(), last_goal_achieved.begin());
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
    hindsight_nb_destination = pt->get<uint>("agent.hindsight_nb_destination");
    gae                     = false;
    update_each_episode = 1;
    
    if(gaussian_policy == 2) {
      double oun_theta = pt->get<double>("agent.noise2");
      double oun_dt = pt->get<double>("agent.noise3");
      oun = new bib::OrnsteinUhlenbeckNoise<double>(this->nb_motors, noise, oun_theta, oun_dt);
    } else if (gaussian_policy == 3 || gaussian_policy == 5){
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
  
    LOG_INFO("dimensionality of NN " << nb_sensors << " (in) " << this->nb_motors << " (out).");
    ann = new NN(nb_sensors, *hidden_unit_a, this->nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type, 0, true, momentum);
    ann_noblob = new NN(*ann, false, ::caffe::Phase::TEST);

#ifdef PARALLEL_INTERACTION
    std::vector<double> weights(ann->number_of_parameters(false), 0.f);
    if (world.rank() == 0)
      ann->copyWeightsTo(weights.data(), false);

    broadcast(world, weights, 0);
    if (world.rank() != 0)
      ann->copyWeightsFrom(weights.data(), false);
#endif

#ifdef PARALLEL_INTERACTION
    if (world.rank() == 0)
      vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, 0, false, momentum);
#else
    vnn = new NN(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, 0, false, momentum);
#endif
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
    ann_noblob->copyWeightsFrom(weights, false);
    delete[] weights;
  }

  void update_critic(const caffe::Blob<double>& all_states, const caffe::Blob<double>& all_next_states,
    const caffe::Blob<double>& r_gamma_coef) {
    
    if (trajectory.size() > 0) {
      caffe::Blob<double> v_target(trajectory.size(), 1, 1, 1);

      //remove trace of old policy
      auto iter = [&]() {
        auto all_nextV = vnn->computeOutVFBlob(all_next_states, empty_action);
        auto all_V = vnn->computeOutVFBlob(all_states, empty_action);
        //all_V must be computed after all_nextV to use learn_blob_no_full_forward

//#ifdef CAFFE_CPU_ONLY
        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), v_target.mutable_cpu_data());
        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), v_target.cpu_data(), v_target.mutable_cpu_data());

        double *pv_target = v_target.mutable_cpu_data();
        double min_ = - (1.f/(1.f-this->gamma));
        for(int i=0;i<trajectory.size();i++){
            if(pv_target[i] > 0.0)
                pv_target[i] = 0.f;
            else if (pv_target[i] < min_)
                pv_target[i] = min_;
        }

        caffe::caffe_sub(trajectory.size(), v_target.cpu_data(), all_V->cpu_data(), v_target.mutable_cpu_data());
//#else
//      switch (caffe::Caffe::mode()) {
//      case caffe::Caffe::CPU:
//        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), v_target.mutable_cpu_data());
//        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), v_target.cpu_data(), v_target.mutable_cpu_data());
//        caffe::caffe_sub(trajectory.size(), v_target.cpu_data(), all_V->cpu_data(), v_target.mutable_cpu_data());
//        break;
//      case caffe::Caffe::GPU:
//        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), v_target.mutable_gpu_data());
//        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), v_target.gpu_data(), v_target.mutable_gpu_data());
//        caffe::caffe_gpu_sub(trajectory.size(), v_target.gpu_data(), all_V->gpu_data(), v_target.mutable_gpu_data());
//        break;
//      }
//#endif
        
//     Simple computation for lambda return
//    move v_target from GPU to CPU
        double* pdiff = v_target.mutable_cpu_diff();
        const double* pvtarget = v_target.cpu_data();
        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it = trajectory.rbegin(); it != trajectory.rend(); it++) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              index_ep--;
          }
//          if(pvtarget[li] + all_V->cpu_data()[li] > 0.0001f || pvtarget[li] + all_V->cpu_data()[li] < -50.0001f)
//            LOG_DEBUG("MIGHT BE PROBLEMATIC " << (pvtarget[li]+all_V->cpu_data()[li]) << " " << r_gamma_coef.cpu_diff()[li] << " " << all_nextV->cpu_data()[li] << " " << r_gamma_coef.cpu_data()[li]);
          
          if (it->artificial) {
            pdiff[li] = pvtarget[li] * std::min(it->prob, pbar) + prev_delta * std::min(it->prob, cbar);
            prev_delta = this->gamma * lambda * pdiff[li];
          } else {
            pdiff[li] = pvtarget[li] + prev_delta;
            prev_delta = this->gamma * lambda * pdiff[li];
          }
          --li;
        }
//        ASSERT(pdiff[trajectory.size() -1] == pvtarget[trajectory.size() -1] * std::min(trajectory[trajectory.size() - 1].prob, pbar), "pb lambda");
        
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

  void end_episode(bool learning) override {
//     LOG_FILE("policy_exploration", ann->hash());
    if(!learning){
      return;
    }
    
    //learning phase
    if (trajectory_end_points.size() == 0 || trajectory_end_points.back() != trajectory.size())
      trajectory_end_points.push_back(trajectory.size());
    if (episode % update_each_episode != 0)
      return;

//     
// Remove junk data
// 
    std::deque<double> varsums(trajectory_end_points.size(), 0.f);
    for (int traj = trajectory_end_points.size() - 1 ; traj >= 0 ; traj--) {
      int beg = traj == 0 ? 0 : trajectory_end_points[traj-1];
      int end = trajectory_end_points[traj];
      if (end - beg > 1) {
        for (int goal_dim=0; goal_dim < goal_size; goal_dim++) {
          std::function<double(const sample&)> get = [goal_dim](const sample&  s) {
            return s.goal_achieved_unnormed[goal_dim];
          };
          varsums[traj] += bib::Utils::variance<>(trajectory.cbegin() + beg, trajectory.cbegin() + end, get);
        }
      }
      
      //goal_achieved hasn't change at all during the trajectory
      if (varsums[traj] <= 1e-8) {
        // remove already achieved task
//        if (trajectory[beg].r >= -0.0001) {
//          trajectory.erase(trajectory.begin() + beg, trajectory.begin() + end);
//          for (uint i=traj+1;i< trajectory_end_points.size(); i++)
//            trajectory_end_points[i] -= (end - beg);
//          trajectory_end_points.erase(trajectory_end_points.begin() + traj);
//          varsums.erase(varsums.begin() + traj);
//        }
//or tag
        for (auto it = trajectory.begin() + beg; it != trajectory.begin() + end; it++)
            it->interest=false;
      }
    }
 
#ifdef PARALLEL_INTERACTION
    if (world.rank() == 0) {
      trajectory_end_points.clear();
      varsums.clear();
      
      std::vector<std::deque<sample>> all_traj;
      std::vector<std::deque<int>> all_traj_ep;
      std::vector<std::deque<double>> all_varsums;
      gather(world, trajectory, all_traj, 0);
      gather(world, trajectory_end_points, all_traj_ep, 0);
      gather(world, varsums, all_varsums, 0);

      ASSERT(all_traj.size() == all_traj_ep.size(), "pb");
      for (auto it : all_varsums)
        for (auto it2 : it)
            varsums.push_back(it2);
      for (int i=0; i < all_traj.size() ; i++) {
        for (auto d2 : all_traj_ep[i])
          trajectory_end_points.push_back(trajectory.size() + d2);
        
        trajectory.insert(trajectory.end(), all_traj[i].begin(), all_traj[i].end());
      }
    } else {
      gather(world, trajectory, 0);
      gather(world, trajectory_end_points, 0);
      gather(world, varsums, 0);
    }
    
    if (world.rank() == 0) {
#endif
    
//     LOG_DEBUG("#############");
//     for (int i=0;i<trajectory.size(); i++) {
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].goal_achieved);
//       LOG_DEBUG(trajectory[i].r << " " <<i);
//     }
//     bib::Logger::PRINT_ELEMENTS(trajectory_end_points);
//     LOG_DEBUG("#############");
//     LOG_DEBUG("#############");
    
//     
//    update norm on batch
//     
    for (int i=0;i<trajectory.size(); i++) {
      normalizer.update_batch_clip_before(trajectory[i].s, goal_size);//ignore fixed goal in update
      normalizer.update_batch_clip_before(trajectory[i].goal_achieved);
    }
    
#ifdef PARALLEL_INTERACTION
    }
    
//     synchronize normalizer
    bib::OnlineNormalizer on(this->nb_sensors);
    if (world.rank() == 0)
      on.copyFrom(normalizer);
    
    broadcast(world, on, 0);
    
    if (world.rank() != 0)
      normalizer.copyFrom(on);
    
    if (world.rank() == 0) {
#endif
   
    if (trajectory.size() == 0) {
      LOG_INFO("no data left");
      nb_sample_update = 0;
      ASSERT(trajectory_end_points.size() == 0, "");
      return;
    }
    
//     LOG_DEBUG("#############");
//     for (int i=0;i<trajectory.size(); i++) {
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].goal_achieved);
//       LOG_DEBUG(trajectory[i].r << " " <<i);
// //       if(trajectory[i].r >= 0 ){
// //         bib::Logger::PRINT_ELEMENTS(trajectory[i].goal_achieved, ("HERE "+std::to_string(i)+" ").c_str());
// //       }
//     }
// //     bib::Logger::PRINT_ELEMENTS(trajectory_end_points);
// //     exit(1);
    
//     
//  perform norm on batch
// 
     for (int i=0;i<trajectory.size(); i++) {
       std::vector<double> normed_sensors(nb_sensors);
       std::vector<double> normed_goal_size(goal_size);
       std::vector<double> normed_next_s(nb_sensors);
       normalizer.transform_with_double_clip(normed_sensors, trajectory[i].s, false);
       normalizer.transform_with_double_clip(normed_goal_size, trajectory[i].goal_achieved, false);
       normalizer.transform_with_double_clip(normed_next_s, trajectory[i].next_s, false);
       
       std::copy(normed_sensors.begin(), normed_sensors.end(), trajectory[i].s.begin());
       std::copy(normed_goal_size.begin(), normed_goal_size.end(), trajectory[i].goal_achieved.begin());
       std::copy(normed_next_s.begin(), normed_next_s.end(), trajectory[i].next_s.begin());
     }
    
//     LOG_DEBUG("#############");
//     for (int i=0;i<trajectory.size(); i++) {
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].s);
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].goal_achieved);
//       LOG_DEBUG(trajectory[i].r << " " <<i);
//     }
//     exit(1);
    
// 
// data augmentation part
//
    int saved_trajsize=trajectory.size();
 
//   for(int i=0;i < saved_trajsize; i++) {
////      don't generate trajectory where goal achieved hasn't changed
//     if (varsums[i] <= 1e-8)
//       continue;
//       
//      int min_index=0;
//      if(i>0)
//        min_index=trajectory_end_points[i-1];
//
//      for(int j=0;j<hindsight_nb_destination;j++) {
//          uint destination = bib::Seed::unifRandInt(min_index, trajectory_end_points[i]-1);
//
//          for(int k=min_index;k<=destination;k++) {
//            sample sa = trajectory[k];
//            trajectory.push_back(sa);
//            trajectory.back().artificial = true;
//            std::copy(trajectory[destination].goal_achieved.begin(), 
//                  trajectory[destination].goal_achieved.end(), 
//                  trajectory.back().s.begin() + goal_start);
//            std::copy(trajectory[destination].goal_achieved.begin(), 
//                  trajectory[destination].goal_achieved.end(), 
//                  trajectory.back().next_s.begin() + goal_start);
//            
//// //           sparse reward
////              if ( goal_achieved_reward(sa.goal_achieved_unnormed, trajectory[destination].goal_achieved_unnormed)){
////                trajectory.back().r = 0.f;
////                trajectory.back().goal_reached = true;
////                trajectory_end_points.push_back(trajectory.size());
////              }
//// //           --
//
////           dense reward
//            trajectory.back().r = dense_reward(sa.goal_achieved_unnormed, trajectory[destination].goal_achieved_unnormed,
//                                                    sa.next_goal_achieved_unnormed, trajectory[destination].next_goal_achieved_unnormed,
//                                                    sa.s, sa.next_s);
//            if ( trajectory.back().r >= -0.0000001 ) {
//                trajectory_end_points.push_back(trajectory.size());
//                trajectory.back().goal_reached = true;
//            }
////          --
//
//            //should remove junk data after data
//          }
//          trajectory_end_points.push_back(trajectory.size());
//      }
//   }
   
// 
//  compute importance sampling ratio on artificial data
// 
    int artificial_data_size = trajectory.size() - saved_trajsize;
    if (artificial_data_size > 0) {
      caffe::Blob<double> all_states(artificial_data_size, nb_sensors, 1, 1);
      double* pall_states = all_states.mutable_cpu_data();
      
      int li=0;
      for (int i = saved_trajsize; i < trajectory.size(); i++) {
        std::copy(trajectory[i].s.begin(), trajectory[i].s.end(), pall_states + li * nb_sensors);
        li++;
      }
      
      ann->increase_batchsize(artificial_data_size);
      auto ac_out = ann->computeOutBlob(all_states);
      li=0;
      for (int i = saved_trajsize; i < trajectory.size(); i++) {
        trajectory[i].prob = bib::Proba<double>::truncatedGaussianDensity(trajectory[i].a, ac_out->cpu_data(), noise, li * this->nb_motors) / trajectory[i].prob;
        li++;
      }
      delete ac_out;
    }
    
//     LOG_DEBUG("#############");
//     for (int i=0;i<trajectory.size(); i++){
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].s, trajectory[i].artificial ? "arti " : "real ");
//       bib::Logger::PRINT_ELEMENTS(trajectory[i].goal_achieved);
//       LOG_DEBUG(trajectory[i].r << " " <<i);
//     }
//     LOG_DEBUG("#############");
//     LOG_DEBUG("#############");
//     LOG_DEBUG("#############");
//     exit(1);
        
    if(trajectory.size() > 0)
      vnn->increase_batchsize(trajectory.size());
    
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

    update_critic(all_states, all_next_states, r_gamma_coef);

    if (trajectory.size() > 0) {
      const std::vector<double> disable_back_ac(this->nb_motors, 0.00f);
      caffe::Blob<double> deltas(trajectory.size(), 1, 1, 1);

      auto all_nextV = vnn->computeOutVFBlob(all_next_states, empty_action);
      auto all_mine = vnn->computeOutVFBlob(all_states, empty_action);
     
//#ifdef CAFFE_CPU_ONLY
      caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), deltas.mutable_cpu_data());
      caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltas.cpu_data(), deltas.mutable_cpu_data());

      double *pv_target = deltas.mutable_cpu_data();
     double min_ = - (1.f/(1.f-this->gamma));
     for(int i=0;i<trajectory.size();i++){
         if(pv_target[i] > 0.0)
             pv_target[i] = 0.f;
         else if (pv_target[i] < min_)
             pv_target[i] = min_;
     }


      caffe::caffe_sub(trajectory.size(), deltas.cpu_data(), all_mine->cpu_data(), deltas.mutable_cpu_data());
//#else
//      switch (caffe::Caffe::mode()) {
//      case caffe::Caffe::CPU:
//        caffe::caffe_mul(trajectory.size(), r_gamma_coef.cpu_diff(), all_nextV->cpu_data(), deltas.mutable_cpu_data());
//        caffe::caffe_add(trajectory.size(), r_gamma_coef.cpu_data(), deltas.cpu_data(), deltas.mutable_cpu_data());
//        caffe::caffe_sub(trajectory.size(), deltas.cpu_data(), all_mine->cpu_data(), deltas.mutable_cpu_data());
//        break;
//      case caffe::Caffe::GPU:
//        caffe::caffe_gpu_mul(trajectory.size(), r_gamma_coef.gpu_diff(), all_nextV->gpu_data(), deltas.mutable_gpu_data());
//        caffe::caffe_gpu_add(trajectory.size(), r_gamma_coef.gpu_data(), deltas.gpu_data(), deltas.mutable_gpu_data());
//        caffe::caffe_gpu_sub(trajectory.size(), deltas.gpu_data(), all_mine->gpu_data(), deltas.mutable_gpu_data());
//        break;
//      }
//#endif
 
      if(gae){
        //        Simple computation for lambda return
        //        move deltas from GPU to CPU
        double * diff = deltas.mutable_cpu_diff();
        const double* pdeltas = deltas.cpu_data();
        int li=trajectory.size() - 1;
        double prev_delta = 0.;
        int index_ep = trajectory_end_points.size() - 1;
        for (auto it = trajectory.rbegin(); it != trajectory.rend(); it++) {
          if (index_ep >= 0 && trajectory_end_points[index_ep] - 1 == li){
              prev_delta = 0.;
              index_ep--;
          }
          
          if(it->artificial) {
            diff[li] = pdeltas[li] * std::min(it->prob, pbar) + prev_delta * std::min(it->prob, cbar);
            prev_delta = this->gamma * lambda * diff[li];
          } else {
            diff[li] = pdeltas[li] + prev_delta;
            prev_delta = this->gamma * lambda * diff[li];
          }
          --li;
        }
//        ASSERT(diff[trajectory.size() -1] == pdeltas[trajectory.size() -1] * std::min(trajectory[trajectory.size() - 1].prob, pbar), "pb lambda");

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
        if(pdeltas[li] > 0. && it->interest) {
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
      int number_non_artificial_sample = 0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        std::copy(it->pure_a.begin(), it->pure_a.end(), ptarget_treg + li * this->nb_motors);
        if(ignore_poss_ac && pdeltas[li] > 0. || it->artificial) {
            std::copy(disable_back_ac.begin(), disable_back_ac.end(), pdisable_back_treg + li * this->nb_motors);
        }
        if (! it->artificial)
          number_non_artificial_sample++;
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
        ann->increase_batchsize(trajectory.size());
        for(uint sia = 0; sia < stoch_iter_actor; sia++){
          //learn BN
          auto ac_out = ann->computeOutBlob(all_states);
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
            caffe::caffe_mul(size_cost_cacla, target_treg.cpu_diff(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_treg.cpu_data(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            l2distance = caffe::caffe_cpu_asum(size_cost_cacla, diff_treg.cpu_data());
#else
          switch (caffe::Caffe::mode()) {
          case caffe::Caffe::CPU:
            caffe::caffe_sub(size_cost_cacla, target_treg.cpu_data(), ac_out->cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, target_treg.cpu_diff(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            caffe::caffe_mul(size_cost_cacla, diff_treg.cpu_data(), diff_treg.cpu_data(), diff_treg.mutable_cpu_data());
            l2distance = caffe::caffe_cpu_asum(size_cost_cacla, diff_treg.cpu_data());
            break;
          case caffe::Caffe::GPU:
            caffe::caffe_gpu_sub(size_cost_cacla, target_treg.gpu_data(), ac_out->gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, target_treg.gpu_diff(), diff_treg.gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_mul(size_cost_cacla, diff_treg.gpu_data(), diff_treg.gpu_data(), diff_treg.mutable_gpu_data());
            caffe::caffe_gpu_asum(size_cost_cacla, diff_treg.gpu_data(), &l2distance);
            break;
          }
#endif
            l2distance = std::sqrt(l2distance/((double) number_non_artificial_sample*this->nb_motors));

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
      }
      
      conserved_beta = beta;
      if (number_effective_actor_update != 0)
        mean_beta /= (double) number_effective_actor_update;

      delete all_nextV;
      delete all_mine;
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
        bib::XMLEngine::save<>(normalizer, "normalizer", path+".normalizer.data");
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
#ifndef PARALLEL_INTERACTION
    vnn->load(path+".critic");
#else
    if (world.rank() == 0)
      vnn->load(path+".critic");
#endif
    bib::XMLEngine::load<>(normalizer, "normalizer", path+".normalizer.data");
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
  
  uint getGoalSize(){
    return goal_size;
  }

  bool sparse_reward(const std::vector<double>&  a, const std::vector<double>&  b) {
    double sum = 0.f;
    for (int i=0;i<a.size();i++){
      double diff = a[i] - b[i];
      sum += diff*diff;
    }
    sum = std::sqrt(sum);
    return sum < 0.05f;
  };
 
  double dense_reward(const std::vector<double>&  goal_achieved, const std::vector<double>&  desired_goal, 
                               const std::vector<double>&  next_goal_achieved, const std::vector<double>&  next_desired_goal, 
                               const std::vector<double>&  observation, const std::vector<double>&  next_observation) {
     
     if (sparse_reward(goal_achieved, desired_goal))
        return 0.f;

     std::vector<double> mid_goal(3, 0.f);
     mid_goal[0] = next_goal_achieved[0] - (next_desired_goal[0] - next_goal_achieved[0] > 0.f ? 1.f : -1.f)*0.04f;
     mid_goal[1] = next_goal_achieved[1] - (next_desired_goal[1] - next_goal_achieved[1] > 0.f ? 1.f : -1.f)*0.04f;
     mid_goal[2] = next_goal_achieved[2] + 0.07f;
     
     double dist_obj_hand;
     {
       std::vector<double> diff(3, 0.f);
       std::transform(mid_goal.begin(), mid_goal.end(), observation.begin() + 3, diff.begin(), std::minus<double>());
       dist_obj_hand = bib::Utils::euclidien_dist_ref(diff, 0.f)*3.f;
     }
     {
       std::vector<double> diff(3, 0.f);
       std::transform(mid_goal.begin(), mid_goal.end(), next_observation.begin() + 3, diff.begin(), std::minus<double>());
       dist_obj_hand -= bib::Utils::euclidien_dist_ref(diff, 0.f)*3.f;
     }
 
     double dist_goal;
     {
       std::vector<double> diff(3, 0.f);
       std::transform(goal_achieved.begin(), goal_achieved.end(), desired_goal.begin(), diff.begin(), std::minus<double>());
       dist_goal = bib::Utils::euclidien_dist_ref(diff, 0.f)*3.f;
     }
     {
       std::vector<double> diff(3, 0.f);
       std::transform(next_goal_achieved.begin(), next_goal_achieved.end(), next_desired_goal.begin(), diff.begin(), std::minus<double>());
       dist_goal -= bib::Utils::euclidien_dist_ref(diff, 0.f)*3.f;
     }
     if(dist_goal < 0.00005 and dist_goal >= 0.000000001)
        dist_goal = 0.f;

     double r = dist_obj_hand + 100*dist_goal;
     if (r > 0.5)
        r = 0.5;
     else if (r < -0.5)
        r = -0.5;

     return -1. + 0.5 + r;
   }
 

#ifdef PARALLEL_INTERACTION
  int getMPIrank() {
    return world.rank();
  }
#endif

 protected:
#ifndef PARALLEL_INTERACTION
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
#else
  void _dump(std::ostream& out) const override {
    out << std::setw(25) << std::fixed << std::setprecision(22) << this->sum_weighted_reward/this->gamma << " " << 
    this->sum_reward << " " << std::setw(8) << std::fixed << std::setprecision(5) << (world.rank() == 0 ? vnn->error() : 0) << " " << 
    nb_sample_update << " " << std::setprecision(3) << ratio_valid_advantage << " " << std::setprecision(10) << 
    mean_beta << " " << conserved_l2dist << " " << std::setprecision(3) << (world.rank() == 0 ?  vnn->weight_l1_norm() : 0) << " " << 
    ann->weight_l1_norm(true) << " " << std::setprecision(6)  << posdelta_mean << " " << number_effective_actor_update;
  }
  
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << this->sum_weighted_reward/this->gamma << " " << this->sum_reward << 
        " " << std::setw(8) << std::fixed << std::setprecision(5) << (world.rank() == 0 ? vnn->error() : 0) << " " << noise << " " << nb_sample_update <<
          " " << std::setprecision(3) << ratio_valid_advantage << " " << (world.rank() == 0 ?  vnn->weight_l1_norm() : 0) << " " << ann->weight_l1_norm(true);
  }
#endif
  
 private:
  uint nb_sensors;
  uint episode = 1;
  uint step = 0;

  double noise, noise2, noise3;
  uint gaussian_policy;
  bool gae, ignore_poss_ac, conserve_beta, disable_trust_region, disable_cac;
  uint number_fitted_iteration, stoch_iter_actor, stoch_iter_critic;
  uint actor_output_layer_type, hidden_layer_type, momentum;
  double lambda, beta_target;
  double conserved_beta= 0.0001f;
  double mean_beta= 0.f;
  double conserved_l2dist= 0.f;
  int number_effective_actor_update = 0;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  std::vector<double> last_goal_achieved;
  double alpha_v, alpha_a;

  std::deque<sample> trajectory;
  std::deque<int> trajectory_end_points;

  NN* ann;
  NN* ann_noblob;
  NN* vnn = nullptr;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  caffe::Blob<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  float ratio_valid_advantage=0;
  int nb_sample_update = 0;
  double posdelta_mean = 0;
  
  //hindsight
  uint goal_size;
  uint goal_start;
  uint hindsight_nb_destination;
  
  //v trace
  double pbar = 1;
  double cbar = 1;
  
  bib::OnlineNormalizer normalizer;

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

