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


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
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
    try {
      check_feasible = pt->get<bool>("agent.check_feasible");
    } catch(boost::exception const& ) {
    }
    
    try {
      ignore_null_lr = pt->get<bool>("agent.ignore_null_lr");
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
    if(terminate)
      LOG_INFO("mismatch "<< terminate);
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
        parameters = cmaes_GetPtr(evo, "xbestever");
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
        parameters = cmaes_GetPtr(evo, "xbestever");
      
      loadPolicyParameters(parameters);
    }
    
    episode++;
    //LOG_FILE("policy_exploration", ann->hash());
  }
  
  void restoreBest() override {
    const double* parameters = cmaes_GetPtr(evo, "xbestever");
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
      const double* parameters = cmaes_GetPtr(evo, "xbestever");
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
  
private:  
  std::vector<uint>* hidden_unit_a;
  uint nb_sensors;
  double policy_stochasticity;
  uint population, actor_hidden_layer_type, actor_output_layer_type, batch_norm;
  bool gaussian_policy;

  std::shared_ptr<std::vector<double>> last_action;
  
  std::list<double> scores;

  MLP* ann;
  cmaes_t* evo;
  double *const *pop;
  double *arFunvals;
  bool justLoaded = false;
  bool cmaes_UpdateDistribution_done_once = false;
  uint current_individual;
  uint episode;
  double initial_deviation;
  bool check_feasible;
  bool ignore_null_lr;
};

#endif

