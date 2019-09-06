#ifndef PENNFAC_HPP
#define PENNFAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <caffe/util/math_functions.hpp>

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "lapacke.h"

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
    
    delete correlation_matrix;
    
    if( correlation_history > 0) {
      delete ttmg_mu;
      delete ttmg_a;
    }
    
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
      } else if(gaussian_policy == 5 && correlation_history == 0) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianZeroMean(this->nb_motors, noise);
        //assume cholesky upper triangular matrix
        caffe::caffe_cpu_gemv(CblasTrans, this->nb_motors, this->nb_motors, (double)1.f, correlation_matrix->data(), randomized_action->data(), (double)1.f, next_action->data());
        for(int i=0;i<this->nb_motors;i++){
          if(next_action->at(i) > 1.f)
            next_action->at(i) = 1.f;
          else if(next_action->at(i) < -1.f)
            next_action->at(i) = -1.f;
        }
        delete randomized_action;
      } else if(gaussian_policy == 5) {
        if (correlation_history > 1) {
          std::vector<double> tmpcopy(this->nb_motors*correlation_history);
          std::copy(ttmg_mu->begin(), ttmg_mu->end(),  tmpcopy.begin());
          std::copy(tmpcopy.begin(), tmpcopy.end(),  ttmg_mu->begin() + this->nb_motors);
          std::copy(ttmg_a->begin(), ttmg_a->end(),  tmpcopy.begin());
          std::copy(tmpcopy.begin(), tmpcopy.end(),  ttmg_a->begin() + this->nb_motors);
        }
        
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianZeroMean(this->nb_motors, noise);
//         bib::Logger::PRINT_ELEMENTS(*randomized_action, " rand ");
//         bib::Logger::PRINT_ELEMENTS(*next_action, " mu ");
        
        //adapt mu
        if(bib::Utils::rand01() > 0.3) { //cut time depency sometimes in else
          std::vector<double> diff(this->nb_motors*correlation_history);
          caffe::caffe_sub(this->nb_motors * correlation_history, ttmg_a->data(), ttmg_mu->data(), diff.data());
          caffe::caffe_cpu_gemv(CblasNoTrans, this->nb_motors * correlation_history, this->nb_motors, (double) 1.f, sub_mu_matrix->data(), diff.data(), (double) 1.f, next_action->data());
//         bib::Logger::PRINT_ELEMENTS(*next_action, "debug_mu_bar ");
        //assume cholesky upper triangular matrix
          caffe::caffe_cpu_gemv(CblasTrans, this->nb_motors, this->nb_motors, (double)1.f, sub_cov_matrix->data(), randomized_action->data(), (double)1.f, next_action->data());
        } else
          caffe::caffe_cpu_gemv(CblasTrans, this->nb_motors, this->nb_motors, (double)1.f, correlation_matrix->data(), randomized_action->data(), (double)1.f, next_action->data());
//         bib::Logger::PRINT_ELEMENTS(*next_action, " a ");
//         if(step == 1)
//           exit(1);
        
        std::copy(next_action->begin(), next_action->end(),  ttmg_a->begin());
        std::copy(last_pure_action->begin(), last_pure_action->end(),  ttmg_mu->begin());
        
        for(int i=0;i<this->nb_motors;i++){
          if(next_action->at(i) > 1.f)
            next_action->at(i) = 1.f;
          else if(next_action->at(i) < -1.f)
            next_action->at(i) = -1.f;
        }
        delete randomized_action;
      } else if(bib::Utils::rand01() < noise) { //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);

    std::copy(sensors.begin(), sensors.end(), last_state.begin());
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
    correlation_importance = pt->get<double>("agent.correlation_importance");
    correlation_history = pt->get<uint>("agent.correlation_history");
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

#ifdef CAFFE_CPU_ONLY
    LOG_INFO("CPU mode");
    (void) command_args;
#else
    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0){ 
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
    
    ann_testing = new NN(*ann, false, ::caffe::Phase::TEST);
    
    //must be upper triangular matrix
//     std::vector<double> correlation_matrix_hf = {1.375249,  0.238563,  0.015183,  0.290535, -0.315891, -0.437307,  1.047048,  0.408286, -0.078809,  0.156291, -0.348032, -0.427579,  0.238563,  0.786787, -0.261722, -0.171231, -0.175322, -0.065455,  0.168597,  0.328537, -0.144833, -0.144146, -0.197819, -0.117765,  0.015183, -0.261722,  0.713452,  0.101106,  0.049957, -0.029062,  0.107643, -0.192083,  0.225718,  0.102632,  0.066414, -0.003025,  0.290535, -0.171231,  0.101106,  0.775855, -0.009837, -0.264952,  0.245867, -0.076778,  0.128576,  0.148137,  0.105445, -0.135151, -0.315891, -0.175322,  0.049957, -0.009837,  0.592699,  0.14798 , -0.203438, -0.162062,  0.021712,  0.114982,  0.132216,  0.108126, -0.437307, -0.065455, -0.029062, -0.264952,  0.14798 ,  0.663387, -0.37204 , -0.074353, -0.010163, -0.169884,  0.166276,  0.194065,  1.047048,  0.168597,  0.107643,  0.245867, -0.203438, -0.37204 ,  1.375311,  0.23772 ,  0.013584,  0.292078, -0.315952, -0.434919,  0.408286,  0.328537, -0.192083, -0.076778, -0.162062, -0.074353,  0.23772 ,  0.786991, -0.261524, -0.172745, -0.17612 , -0.065683, -0.078809, -0.144833,  0.225718,  0.128576,  0.021712, -0.010163,  0.013584, -0.261524,  0.713465,  0.098562,  0.048416, -0.028952,  0.156291, -0.144146,  0.102632,  0.148137,  0.114982, -0.169884,  0.292078, -0.172745,  0.098562,  0.780596, -0.008542, -0.261269, -0.348032, -0.197819,  0.066414,  0.105445,  0.132216,  0.166276, -0.315952, -0.17612 ,  0.048416, -0.008542,  0.592523,  0.150291, -0.427579, -0.117765, -0.003025, -0.135151,  0.108126,  0.194065, -0.434919, -0.065683, -0.028952, -0.261269,  0.150291,  0.663028};
    
//     H0 cholesky 
    std::vector<double> correlation_matrix_hf = { 1.172546,  0.202825,  0.01224 ,  0.249332, -0.268523, -0.371637,  0.      ,  0.863294, -0.30518 , -0.258251, -0.140362,  0.010726,  0.      ,  0.      ,  0.787604,  0.021766,  0.012234, -0.028352,  0.      ,  0.      ,  0.      ,  0.806803,  0.027837, -0.20554 ,  0.      ,  0.      ,  0.      ,  0.      ,  0.707185,  0.080117,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.690415};
    
    std::vector<double> upper_left_hf = {1.375249,  0.238563,  0.015183,  0.290535, -0.315891, -0.437307,  0.238563,  0.786787, -0.261722, -0.171231, -0.175322, -0.065455,  0.015183, -0.261722,  0.713452,  0.101106,  0.049957, -0.029062,  0.290535, -0.171231,  0.101106,  0.775855, -0.009837, -0.264952, -0.315891, -0.175322,  0.049957, -0.009837,  0.592699,  0.14798 , -0.437307, -0.065455, -0.029062, -0.264952,  0.14798 ,  0.663387};
    std::vector<double> upper_right_hf = {1.047048,  0.408286, -0.078809,  0.156291, -0.348032, -0.427579,  0.168597,  0.328537, -0.144833, -0.144146, -0.197819, -0.117765,  0.107643, -0.192083,  0.225718,  0.102632,  0.066414, -0.003025,  0.245867, -0.076778,  0.128576,  0.148137,  0.105445, -0.135151, -0.203438, -0.162062,  0.021712,  0.114982,  0.132216,  0.108126, -0.37204 , -0.074353, -0.010163, -0.169884,  0.166276,  0.194065};
    std::vector<double> bottom_left_hf = {1.047048,  0.168597,  0.107643,  0.245867, -0.203438, -0.37204 ,  0.408286,  0.328537, -0.192083, -0.076778, -0.162062, -0.074353, -0.078809, -0.144833,  0.225718,  0.128576,  0.021712, -0.010163,  0.156291, -0.144146,  0.102632,  0.148137,  0.114982, -0.169884, -0.348032, -0.197819,  0.066414,  0.105445,  0.132216,  0.166276, -0.427579, -0.117765, -0.003025, -0.135151,  0.108126,  0.194065};
    std::vector<double> inv_bottom_right_hf = {1.076633, -0.299715, -0.095462, -0.29491 ,  0.368688,  0.472586, -0.299715,  1.721995,  0.558192,  0.465147,  0.283334,  0.117431, -0.095462,  0.558192,  1.616162, -0.024783, -0.03286 ,  0.060932, -0.29491 ,  0.465147, -0.024783,  1.677507, -0.130292,  0.542112,  0.368688,  0.283334, -0.03286 , -0.130292,  2.031015, -0.243243,  0.472586,  0.117431,  0.060932,  0.542112, -0.243243,  2.101282};
    
    // cholesky(UL - UR @ IBR @ BL)
    std::vector<double> sub_cov_matrix_hf = {0.693829, -0.016779,  0.020195,  0.216119, -0.146121, -0.188821,  0.      ,  0.771047, -0.208573, -0.150151, -0.08068 , -0.0099  ,  0.      ,  0.      ,  0.745927, -0.026867,  0.009598, -0.012033,  0.      ,  0.      ,  0.      ,  0.762361,  0.011613, -0.222311,  0.      ,  0.      ,  0.      ,  0.      ,  0.687421,  0.096491,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.660971};
    
    // UR @ IBR
    std::vector<double> sub_mu_matrix_hf = {0.635967,  0.269137, -0.017911, -0.04119 , -0.118911, -0.191117,  0.010798,  0.297436, -0.063885, -0.173187, -0.194343, -0.168051,  0.144704, -0.170834,  0.242392,  0.035187,  0.100098,  0.075194,  0.206764, -0.051221,  0.1261  ,  0.050088,  0.292403, -0.11432 , -0.106592, -0.102335, -0.036557,  0.218347,  0.105614,  0.143525, -0.174179, -0.031323, -0.011841, -0.126056,  0.154739,  0.090072};

//     correlation_matrix = new std::vector<double>(this->nb_motors*this->nb_motors);
//     if (correlation_history == 0) {
      correlation_matrix = new std::vector<double>((this->nb_motors*(correlation_history))*(this->nb_motors*(correlation_history)));
      ASSERT(correlation_matrix_hf.size() == correlation_matrix->size(), "pb size " << correlation_matrix_hf.size() << " " <<  correlation_matrix->size());
      std::copy(correlation_matrix_hf.begin(), correlation_matrix_hf.end(), correlation_matrix->begin());
//     }
    
//     upper_left = new std::vector<double>(this->nb_motors*this->nb_motors);
//     ASSERT(upper_left_hf.size() == upper_left->size(), "pb size ");
//     std::copy(upper_left_hf.begin(), upper_left_hf.end(), upper_left->begin());
//     
//     upper_right = new std::vector<double>((this->nb_motors*(correlation_history))*(this->nb_motors*(correlation_history)));
//     ASSERT(upper_right_hf.size() == upper_right->size(), "pb size ");
//     std::copy(upper_right_hf.begin(), upper_right_hf.end(), upper_right->begin());
//     
//     bottom_left = new std::vector<double>((this->nb_motors*(correlation_history))*(this->nb_motors*(correlation_history)));
//     ASSERT(bottom_left_hf.size() == bottom_left->size(), "pb size ");
//     std::copy(bottom_left_hf.begin(), bottom_left.end(), bottom_left->begin());
//     
//     inv_bottom_right = new std::vector<double>((this->nb_motors*(correlation_history))*(this->nb_motors*(correlation_history)));
//     ASSERT(inv_bottom_right_hf.size() == inv_bottom_right->size(), "pb size ");
//     std::copy(inv_bottom_right_hf.begin(), inv_bottom_right_hf.end(), inv_bottom_right->begin());
    
    sub_cov_matrix = new std::vector<double>(this->nb_motors*this->nb_motors);
    ASSERT(sub_cov_matrix_hf.size() == sub_cov_matrix->size(), "pb size ");
    std::copy(sub_cov_matrix_hf.begin(), sub_cov_matrix_hf.end(), sub_cov_matrix->begin());

    sub_mu_matrix = new std::vector<double>(this->nb_motors*this->nb_motors);
    ASSERT(sub_mu_matrix_hf.size() == sub_mu_matrix->size(), "pb size ");
    std::copy(sub_mu_matrix_hf.begin(), sub_mu_matrix_hf.end(), sub_mu_matrix->begin());
    
//     if cholesky needed:
//     LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', this->nb_motors, correlation_matrix->data(), this->nb_motors);
//     bib::Logger::PRINT_ELEMENTS(*correlation_matrix);
    
    if( correlation_history > 0) {
      ttmg_mu = new std::vector<double>(this->nb_motors * correlation_history, 0.f);
      ttmg_a = new std::vector<double>(this->nb_motors * correlation_history, 0.f);
    }
    
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
    
    if(correlation_history > 1) {
      std::fill(ttmg_mu->begin(), ttmg_mu->end(), 0.);
      std::fill(ttmg_a->begin(), ttmg_a->end(), 0.);
    }
    
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
        decltype(vnn_testing->computeOutVFBlob(all_next_states, empty_action)) all_nextV;
        if(batch_norm_critic != 0)
        {
          double* weights = new double[vnn->number_of_parameters(false)];
          vnn->copyWeightsTo(weights, false);
          vnn_testing->copyWeightsFrom(weights, false);
          delete[] weights;
          all_nextV = vnn_testing->computeOutVFBlob(all_next_states, empty_action);
        } else 
          all_nextV = vnn->computeOutVFBlob(all_next_states, empty_action);
        auto all_V = vnn->computeOutVFBlob(all_states, empty_action);

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
    trajectory_end_points.push_back(trajectory.size());
    if (episode % update_each_episode != 0)
      return;

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
          
          //correlation cost
          caffe::Blob<double> target_corr(trajectory.size(), this->nb_motors, 1, 1);
          
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
          
//           caffe::caffe_cpu_gemm(CblasNoTrans, CblasTrans, trajectory.size(), this->nb_motors, this->nb_motors, (double)1.f, ac_out->cpu_data(), correlation_matrix->data(), (double)0., target_corr.mutable_cpu_data());
//           caffe::caffe_sub(size_cost_cacla, target_corr.cpu_data(), ac_out->cpu_data(), target_corr.mutable_cpu_diff());
//           caffe::caffe_scal(size_cost_cacla, correlation_importance, target_corr.mutable_cpu_diff());
//           
//           caffe::caffe_add(size_cost_cacla, target_corr.cpu_diff(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
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
            
//             caffe::caffe_cpu_gemm(CblasNoTrans, CblasTrans, trajectory.size(), this->nb_motors, this->nb_motors, (double)1.f, ac_out->cpu_data(), correlation_matrix->data(), (double)0., target_corr.mutable_cpu_data());
//             caffe::caffe_sub(size_cost_cacla, target_corr.cpu_data(), ac_out->cpu_data(), target_corr.mutable_cpu_diff());
//             caffe::caffe_scal(size_cost_cacla, correlation_importance, target_corr.mutable_cpu_diff());
//             
//             caffe::caffe_add(size_cost_cacla, target_corr.cpu_diff(), diff_cac.cpu_data(), diff_cac.mutable_cpu_data());
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
            
//             caffe::caffe_gpu_gemm(CblasNoTrans, CblasTrans, trajectory.size(), this->nb_motors, this->nb_motors, (double)1.f, ac_out->gpu_data(), correlation_matrix->data(), (double)0., target_corr.mutable_gpu_data());
//             caffe::caffe_gpu_sub(size_cost_cacla, target_corr.gpu_data(), ac_out->gpu_data(), target_corr.mutable_gpu_diff());
//             caffe::caffe_gpu_scal(size_cost_cacla, correlation_importance, target_corr.mutable_gpu_diff());
//             
//             caffe::caffe_gpu_add(size_cost_cacla, target_corr.gpu_diff(), diff_cac.gpu_data(), diff_cac.mutable_gpu_data());
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
    
    nb_sample_update= trajectory.size();
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
  bool gae, ignore_poss_ac, conserve_beta, disable_trust_region, disable_cac;
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
  NN* ann_testing_blob;
  NN* vnn_testing;
  std::vector<double>* correlation_matrix;
  std::vector<double> *upper_left, *upper_right, *bottom_left, *inv_bottom_right, *sub_cov_matrix, *sub_mu_matrix;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  caffe::Blob<double> empty_action; //dummy action cause c++ cannot accept null reference
  double bestever_score;
  int update_each_episode;
  bib::OrnsteinUhlenbeckNoise<double>* oun = nullptr;
  float ratio_valid_advantage=0;
  int nb_sample_update = 0;
  double posdelta_mean = 0;
  double correlation_importance;
  uint correlation_history;
  
  //temporal truncated multivariate gaussian
  std::vector<double> *ttmg_mu, *ttmg_a;
  
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

