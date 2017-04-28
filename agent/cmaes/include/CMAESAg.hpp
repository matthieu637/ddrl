#ifndef CMAESAG_HPP
#define CMAESAG_HPP

#include <vector>
#include <string>
#include <type_traits>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/functional/hash.hpp>

#include "arch/ARLAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "nn/MLP.hpp"
#include "nn/DevMLP.hpp"
#include "nn/DODevMLP.hpp"
#include "cmaes_interface.h"

template<typename NN = MLP>
class CMAESAg : public arch::ARLAgent<arch::AgentProgOptions> {
 public:
  CMAESAg(unsigned int _nb_motors, unsigned int _nb_sensors)
  : arch::ARLAgent<arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors){

  }

  virtual ~CMAESAg() {
    cmaes_exit(evo);
    delete evo;
    delete ann;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double , const std::vector<double>& sensors,
                                 bool learning, bool, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, policy_stochasticity);
        delete next_action;
        next_action = randomized_action;
      } else {
        if(bib::Utils::rand01() < policy_stochasticity)
          for (uint i = 0; i < next_action->size(); i++)
            next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    } //TODO: might not be necessary in the future?
    else {
      for (uint i = 0; i < next_action->size(); i++)
        next_action->at(i) = std::min(std::max(next_action->at(i), (double)-1.f), (double) 1.f);
    }
    
    last_action.reset(next_action);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* vm) override {
    hidden_unit_a               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    actor_hidden_layer_type     = pt->get<uint>("agent.actor_hidden_layer_type");
    actor_output_layer_type     = pt->get<uint>("agent.actor_output_layer_type");
    batch_norm                  = pt->get<uint>("agent.batch_norm");
    population                  = pt->get<uint>("agent.population");
    gaussian_policy             = pt->get<bool>("agent.gaussian_policy");
    policy_stochasticity        = pt->get<double>("agent.policy_stochasticity");
    initial_deviation           = pt->get<double>("agent.initial_deviation");
    
    check_feasible = true;
    ignore_null_lr = true;
    racing = false;
    error_count = 0;
    xbestever_score = std::numeric_limits<double>::max();
    
    try {
      check_feasible = pt->get<bool>("agent.check_feasible");
    } catch(boost::exception const& ) {
    }
    
    try {
      ignore_null_lr = pt->get<bool>("agent.ignore_null_lr");
    } catch(boost::exception const& ) {
    }
    
    try {
      racing = pt->get<bool>("agent.racing");
    } catch(boost::exception const& ) {
    }
    
    episode = 0;
    
    ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, 0.1, 1, actor_hidden_layer_type, actor_output_layer_type, batch_norm);
    if(std::is_same<NN, DevMLP>::value)
      ann->exploit(pt, static_cast<CMAESAg *>(old_ag)->ann);
    else if(std::is_same<NN, DODevMLP>::value)
      ann->exploit(pt, nullptr);

//     const uint dimension = (nb_sensors+1)*hidden_unit_a->at(0) + (hidden_unit_a->at(0)+1)*nb_motors;
    const uint dimension = ann->number_of_parameters(ignore_null_lr);
    double* startx  = new double[dimension];
    double* deviation  = new double[dimension];
    for(uint j=0; j< dimension; j++){
      deviation[j] = initial_deviation;
    }
    ann->copyWeightsTo(startx, ignore_null_lr);
    
    evo = new cmaes_t;
    arFunvals = cmaes_init(evo, dimension, startx, deviation, 0, population, NULL/*"config.cmaes.ini"*/);
    delete[] startx;
    delete[] deviation;
//     evo->sp.stopTolFun = 1e-150;
//     evo->sp.stopTolFunHist = 1e-150;
//     evo->sp.stopTolUpXFactor = 1e50;
    
    printf("%s\n", cmaes_SayHello(evo));
    new_population();
    LOG_DEBUG(cmaes_Get(evo, "lambda") << " " << dimension << " " << population << " " << cmaes_Get(evo, "N"));
    if (population < 2)
      LOG_DEBUG("population too small, changed to : " << (4+(int)(3*log((double)dimension))));
    
#ifndef NDEBUG
    if(vm->count("continue") > 0){
      uint continue_save_each          = DEFAULT_AGENT_SAVE_EACH_CONTINUE;
      try {
        continue_save_each            = pt->get<uint>("simulation.continue_save_each");
      } catch(boost::exception const& ) {
      }
      
      if(continue_save_each % (int) cmaes_Get(evo, "lambda") != 0){
        LOG_ERROR("continue_save_each must be a multiple of the population size !");
        exit(1);
      }
    }
#endif
  }
  
  bool is_feasible(const double* parameters){
      for(uint i=0;i < (uint) cmaes_Get(evo, "dim");i++)
        if(fabs(parameters[i]) >= 500.f){
          return false;
        }
        
      return true;
  }
  
  void new_population(){
    const char * terminate =  cmaes_TestForTermination(evo);
    if(terminate){
      LOG_INFO("mismatch "<< terminate);
      error_count++;
      
      if(error_count > 20 && racing){
        LOG_FILE(DEFAULT_END_FILE, "-1");
        exit(0);
      }
    }
    //ASSERT(!cmaes_TestForTermination(evo), "mismatch "<< cmaes_TestForTermination(evo));
    
    current_individual = 0;
    pop = cmaes_SamplePopulation(evo);
    
    if(check_feasible){
      //check that the population is feasible
      bool allfeasible = true;
      for (int i = 0; i < cmaes_Get(evo, "popsize"); ++i)
        while (!is_feasible(pop[i])){
          cmaes_ReSampleSingle(evo, i);
          allfeasible = false;
        }
        
      if(!allfeasible)
        LOG_INFO("non feasible solution produced");
    }
  }

  void start_instance(bool learning) override {
    last_action = nullptr;
    scores.clear();
    
    if(std::is_same<NN, DODevMLP>::value)
      if(static_cast<DODevMLP *>(ann)->inform(episode)){
        LOG_INFO("reset learning catched");
        const double* parameters = nullptr;
        parameters = getBestSolution();
        loadPolicyParameters(parameters);
        
        cmaes_exit(evo);
        delete evo;
        
        const uint dimension = ann->number_of_parameters(ignore_null_lr);
        double* startx  = new double[dimension];
        double* deviation  = new double[dimension];
        for(uint j=0; j< dimension; j++){
          deviation[j] = initial_deviation;
        }
        ann->copyWeightsTo(startx, ignore_null_lr);
        
        evo = new cmaes_t;
        arFunvals = cmaes_init(evo, dimension, startx, deviation, 0, population, NULL/*"config.cmaes.ini"*/);
        delete[] startx;
        delete[] deviation;
        new_population();
      }
    
    if(!justLoaded){
      //put individual into NN
      const double* parameters = nullptr;
      if(learning || !cmaes_UpdateDistribution_done_once)
        parameters = pop[current_individual];
      else
        parameters = getBestSolution();
      
      loadPolicyParameters(parameters);
    }
    
    episode++;
    //LOG_FILE("policy_exploration", ann->hash());
  }
  
  void restoreBest() override {
    const double* parameters = getBestSolution();
    loadPolicyParameters(parameters);
  }

  void end_episode() override {
      scores.push_back(-sum_weighted_reward);
  }
  
  void end_instance(bool learning) override {
    if(learning){
      arFunvals[current_individual] = std::accumulate(scores.begin(), scores.end(), 0.f) / scores.size();
      
      current_individual++;
      if(current_individual >= cmaes_Get(evo, "lambda")){
        cmaes_UpdateDistribution(evo, arFunvals);
        cmaes_UpdateDistribution_done_once=true;
        new_population();
      }
    }
    justLoaded = false;
    
//     LOG_DEBUG(cmaes_Get(evo, "fbestever"));
//     const double* parameters = cmaes_GetPtr(evo, "xbestever");
//     bib::Logger::PRINT_ELEMENTS(parameters, ann->number_of_parameters());
//     LOG_DEBUG("");
//     LOG_DEBUG("");
  }
  
   void write_actionf_file(const std::string& file){
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
//         out << ann->computeOut(x)[0];
        LOG_ERROR("todo");
        out << std::endl;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 6);
      out.close();
/*
      close all; clear all; 
      X=load("ac_func.data");
      tmp_ = X(:,2); X(:,2) = X(:,3); X(:,3) = tmp_;
      key = X(:, 1:2);
      for i=1:size(key, 1)
       subkey = find(sum(X(:, 1:2) == key(i,:), 2) == 2);
       data(end+1, :) = [key(i, :) mean(X(subkey, end))];
      endfor
      [xx,yy] = meshgrid (linspace (-1,1,300));
      griddata(data(:,1), data(:,2), data(:,3), xx, yy, "linear"); xlabel('theta_1'); ylabel('theta_2');
*/    
   }

  void save(const std::string& path, bool save_best) override {
    if(!save_best || !cmaes_UpdateDistribution_done_once){
      ann->save(path+".actor");
    } else {
      NN* to_be_restaured = new NN(*ann, true);
      const double* parameters = getBestSolution();
      loadPolicyParameters(parameters);
      ann->save(path+".actor");
      delete ann;
      ann = to_be_restaured;
    }
  }

  void load(const std::string& path) override {
    justLoaded = true;
    ann->load(path+".actor");
  }
  
  void save_run() override {
    if(current_individual == 1){
      const double* xbptr_ = getBestSolution();
      std::vector<double> xbestever_((int)cmaes_Get(evo, "N"));
      xbestever_.assign(xbptr_, xbptr_ + (int)cmaes_Get(evo, "N"));
      double bs = getBestScore();
      
      struct algo_state st = {scores, justLoaded, cmaes_UpdateDistribution_done_once, 
        current_individual, episode, error_count, bs, xbestever_};
      bib::XMLEngine::save(st, "algo_state", "continue.algo_state.data");
      cmaes_WriteToFile(evo, "resume", "continue.cmaes.data");
    }
  }
  
  void load_previous_run() override {
    auto algo_state_ = bib::XMLEngine::load<struct algo_state>("algo_state", "continue.algo_state.data");
    scores = algo_state_->scores;
    justLoaded = algo_state_->justLoaded;
    cmaes_UpdateDistribution_done_once = algo_state_->cmaes_UpdateDistribution_done_once;
    current_individual = algo_state_->current_individual;
    episode = algo_state_->episode;
    error_count = algo_state_->error_count;
    xbestever_score = algo_state_->xbestever_score;
    xbestever_ptr = algo_state_->xbestever_ptr;
    delete algo_state_;
    char file_[] = "continue.cmaes.data";
    cmaes_resume_distribution(evo, file_);
    new_population();
  }

  MLP* getNN(){
    return ann;
  }
  
 protected:
  void _display(std::ostream& out) const override {
    out << " " << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

  void _dump(std::ostream& out) const override {
    out << " " << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

private:
  void loadPolicyParameters(const double* parameters){
    ann->copyWeightsFrom(parameters, ignore_null_lr);
  }
  
  const double* getBestSolution(){
    if(cmaes_Get(evo, "fbestever") < xbestever_score)
      return cmaes_GetPtr(evo, "xbestever");
    else
      return xbestever_ptr.data();
  }
  
  double getBestScore(){
    return std::min(cmaes_Get(evo, "fbestever"), xbestever_score);
  }
  
private:
  //initilized by constructor
  uint nb_sensors;
  
  //initialized by invoke
  std::vector<uint>* hidden_unit_a;
  double policy_stochasticity;
  uint population, actor_hidden_layer_type, actor_output_layer_type, batch_norm;
  bool gaussian_policy;
  double initial_deviation;
  bool check_feasible;
  bool ignore_null_lr;
  bool racing;
  MLP* ann;
  cmaes_t* evo;
  double *arFunvals;

  //internal mecanisms
  std::shared_ptr<std::vector<double>> last_action;
  std::list<double> scores;
  double *const *pop;
  bool justLoaded = false;
  bool cmaes_UpdateDistribution_done_once = false;
  uint current_individual;
  uint episode;
  uint error_count;
  double xbestever_score;
  std::vector<double> xbestever_ptr;
  
  struct algo_state {
    std::list<double> scores;
    bool justLoaded;
    bool cmaes_UpdateDistribution_done_once;
    uint current_individual;
    uint episode;
    uint error_count;
    double xbestever_score;
    std::vector<double> xbestever_ptr;
    
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int) {
      ar& BOOST_SERIALIZATION_NVP(scores);
      ar& BOOST_SERIALIZATION_NVP(justLoaded);
      ar& BOOST_SERIALIZATION_NVP(cmaes_UpdateDistribution_done_once);
      ar& BOOST_SERIALIZATION_NVP(current_individual);
      ar& BOOST_SERIALIZATION_NVP(episode);
      ar& BOOST_SERIALIZATION_NVP(error_count);
      ar& BOOST_SERIALIZATION_NVP(xbestever_score);
      ar& BOOST_SERIALIZATION_NVP(xbestever_ptr);
    }
  };
};

#endif

