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
      delete ttmg_beta;
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
        caffe::caffe_cpu_gemv(CblasTrans, this->nb_motors, this->nb_motors, (double)1.f, correlation_matrix->data(), randomized_action->data(), (double)1.f, next_action->data());
        for(int i=0;i<this->nb_motors;i++){
          if(next_action->at(i) > 1.f)
            next_action->at(i) = 1.f;
          else if(next_action->at(i) < -1.f)
            next_action->at(i) = -1.f;
        }
        delete randomized_action;
      } else if(gaussian_policy == 5) {
        std::vector<double> tmpcopy(this->nb_motors*correlation_history);
        std::vector<double> tmpout(this->nb_motors*(correlation_history+1));
        std::copy(ttmg_mu->begin(), ttmg_mu->begin()  + (this->nb_motors*correlation_history),  tmpcopy.begin());
        std::copy(tmpcopy.begin(), tmpcopy.end(),  ttmg_mu->begin() + this->nb_motors);
        std::copy(ttmg_beta->begin(), ttmg_beta->begin()  + (this->nb_motors*correlation_history),  tmpcopy.begin());
        std::copy(tmpcopy.begin(), tmpcopy.end(),  ttmg_beta->begin() + this->nb_motors);
        std::copy(ttmg_mu->begin(), ttmg_mu->end(),  tmpout.begin());
        
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianZeroMean(this->nb_motors, noise);
        std::copy(next_action->begin(), next_action->end(),  ttmg_mu->begin());
        std::copy(randomized_action->begin(), randomized_action->end(),  ttmg_beta->begin());
        caffe::caffe_cpu_gemv(CblasTrans, this->nb_motors * (correlation_history+1), this->nb_motors * (correlation_history+1), (double)1.f, 
                              correlation_matrix->data(), ttmg_beta->data(), (double)1.f, tmpout.data());
        
        std::copy(tmpout.begin(), tmpout.begin()  + this->nb_motors,  next_action->begin());
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
    std::vector<double> correlation_matrix_hf = { 1.174186,  0.203411,  0.012921,  0.248043, -0.269374, -0.373182,  0.894234,  0.348071, -0.067611,  0.133598, -0.29758 , -0.365179,  0.775182,  0.403143, -0.136758,  0.017601, -0.306586, -0.295184,  0.581817,  0.442755, -0.186385, -0.104796, -0.304684, -0.21666 ,  0.384656,  0.462858, -0.237002, -0.219181, -0.299025, -0.144615,  0.      ,  0.862825, -0.308543, -0.258294, -0.141422,  0.01231 , -0.014487,  0.298075, -0.153373, -0.200086, -0.161934, -0.050311, -0.077546,  0.174713, -0.026055, -0.130643, -0.148351, -0.087492, -0.124128,  0.080847,  0.048199, -0.074657, -0.131846, -0.078991, -0.207148, -0.037545,  0.136553, -0.042467, -0.070184, -0.045874,  0.      ,  0.      ,  0.787097,  0.023074,  0.011796, -0.026008,  0.116819, -0.133945,  0.228321,  0.04935 ,  0.025144, -0.017432,  0.214212, -0.054297,  0.091971,  0.067139, -0.000265, -0.038273,  0.270156, -0.054468, -0.041571,  0.085952,  0.000596, -0.067899,  0.206938,  0.010235, -0.169479,  0.095322, -0.02856 , -0.094776,  0.      ,  0.      ,  0.      ,  0.805355,  0.024498, -0.209719,  0.022091, -0.103472,  0.125554,  0.075655,  0.168967, -0.072349,  0.050156, -0.077483,  0.11688 ,  0.092764,  0.044356, -0.014   ,  0.128849, -0.048129,  0.107671, -0.025845,  0.014776, -0.01872 ,  0.168651, -0.032637,  0.054602, -0.050323, -0.034873, -0.015527,  0.      ,  0.      ,  0.      ,  0.      ,  0.706358,  0.077181,  0.046136, -0.030253, -0.032914,  0.168428,  0.035216,  0.007171,  0.12418 , -0.002795,  0.025727,  0.130035, -0.004394, -0.090284,  0.17148 ,  0.0394  ,  0.000364,  0.109068, -0.023804, -0.11271 ,  0.229727,  0.067599,  0.039725,  0.087276, -0.067343, -0.144924,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.688788, -0.050369,  0.042815,  0.001872, -0.163852,  0.133731,  0.0633  ,  0.038175,  0.040947, -0.072474,  0.009248,  0.0371  ,  0.056172,  0.110763,  0.11317 , -0.043674,  0.025352, -0.020408, -0.043703,  0.124344,  0.057232, -0.000541,  0.119602, -0.040609, -0.138616,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.747981, -0.062047,  0.060067,  0.193919, -0.073748, -0.140295,  0.433647,  0.082398,  0.032948,  0.165232, -0.102082, -0.205091,  0.468155,  0.119369,  0.008794,  0.133976, -0.118527, -0.191927,  0.405633,  0.140276,  0.013912,  0.087783, -0.113449, -0.152975,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.736018, -0.201222, -0.163734, -0.019424,  0.075468, -0.020466,  0.169955, -0.081086, -0.108818, -0.070216, -0.01723 ,  0.005013,  0.062066,  0.010972, -0.041333, -0.067607, -0.083996,  0.015907,  0.021382,  0.039553,  0.022283, -0.087225, -0.09402 ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.75698 ,  0.025712, -0.029083, -0.031264,  0.086893, -0.115206,  0.211502,  0.037854,  0.011129, -0.025306,  0.161316, -0.032804,  0.09342 ,  0.060346, -0.009783, -0.033927,  0.224226, -0.048221, -0.007985,  0.073163,  0.007211, -0.051977,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.768631,  0.013201, -0.221236,  0.009508, -0.084501,  0.114235,  0.047754,  0.165789, -0.063401,  0.034241, -0.059976,  0.114048,  0.081631,  0.032224, -0.012151,  0.09119 , -0.055699,  0.11256 , -0.019529,  0.017586, -0.021469,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.649448,  0.045834,  0.035615,  0.003673, -0.047159,  0.130328, -0.011238, -0.014062,  0.053801,  0.014307,  0.007983,  0.107205, -0.029978, -0.080625,  0.069029,  0.05095 , -0.014411,  0.060844, -0.017228, -0.074796,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.663382, -0.032742,  0.072824, -0.003892, -0.189299,  0.117536,  0.053642,  0.028538,  0.074365, -0.076868, -0.031642,  0.012208,  0.065469,  0.081109,  0.136662, -0.051135, -0.032084, -0.039386, -0.028442,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.71082 , -0.091297,  0.082399,  0.200436, -0.071692, -0.143935,  0.390972,  0.043817,  0.081302,  0.183362, -0.094461, -0.206023,  0.437813,  0.067162,  0.07541 ,  0.151047, -0.094625, -0.175167,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.717874, -0.185123, -0.139865, -0.005669,  0.080678, -0.011342,  0.132179, -0.05556 , -0.081849, -0.051993, -0.025436,  0.032972,  0.025057,  0.037134, -0.001998, -0.052454, -0.086675,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.747281,  0.020588, -0.023748, -0.019636,  0.082789, -0.100819,  0.188818,  0.033594,  0.015735, -0.012124,  0.156783, -0.012164,  0.064579,  0.057408, -0.005103, -0.024261,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.757448,  0.012995, -0.220248, -0.004157, -0.07528 ,  0.104539,  0.025724,  0.16347 , -0.047375,  0.005405, -0.047587,  0.091219,  0.052449,  0.032305,  0.009441,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.640743,  0.031013,  0.018944,  0.006056, -0.049858,  0.134187, -0.020826, -0.022436,  0.031765,  0.019721,  0.004572,  0.101227, -0.029874, -0.073673,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.653691, -0.043131,  0.079486,  0.002795, -0.18807 ,  0.103634,  0.043586,  0.01121 ,  0.073378, -0.062211, -0.036924,  0.004891,  0.063455,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.690514, -0.092532,  0.090219,  0.194674, -0.078139, -0.135825,  0.355695,  0.048624,  0.08929 ,  0.158619, -0.089115, -0.179161,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.69819 , -0.17725 , -0.127134,  0.004568,  0.079731, -0.012243,  0.102221, -0.050738, -0.067171, -0.039931, -0.020451,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.731947,  0.01627 , -0.015804, -0.009605,  0.07842 , -0.082754,  0.140801,  0.027547,  0.018108, -0.007802,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.740252,  0.016798, -0.206935, -0.011594, -0.062375,  0.084812, -0.017061,  0.160549, -0.024904,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.63498 ,  0.022641,  0.01269 ,  0.010374, -0.036232,  0.136677, -0.028036, -0.027003,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.64572 , -0.040565,  0.076378,  0.019248, -0.174956,  0.097298,  0.025263,  0.      ,  0.      ,  0.      ,  0.      ,
        0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.674863, -0.088506,  0.078228,  0.182292, -0.079042, -0.122494,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.682811, -0.158604, -0.111008,  0.008609,  0.075669,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.703178,  0.003227, -0.012886, -0.003496,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.717137,  0.016111, -0.184827,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.632252,  0.020559,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.      ,  0.640879};

//     correlation_matrix = new std::vector<double>(this->nb_motors*this->nb_motors);
    correlation_matrix = new std::vector<double>((this->nb_motors*(correlation_history+1))*(this->nb_motors*(correlation_history+1)));
    ASSERT(correlation_matrix_hf.size() == correlation_matrix->size(), "pb size " << correlation_matrix_hf.size() << " " <<  correlation_matrix->size());
    std::copy(correlation_matrix_hf.begin(), correlation_matrix_hf.end(), correlation_matrix->begin());
//     if cholesky needed:
//     LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', this->nb_motors, correlation_matrix->data(), this->nb_motors);
//     bib::Logger::PRINT_ELEMENTS(*correlation_matrix);
    
    if( correlation_history > 0) {
      ttmg_mu = new std::vector<double>(this->nb_motors * (correlation_history+1), 0.f);
      ttmg_beta = new std::vector<double>(this->nb_motors * (correlation_history+1), 0.f);
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
      std::fill(ttmg_beta->begin(), ttmg_beta->end(), 0.);
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
  std::vector<double> *ttmg_mu, *ttmg_beta;
  
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

