#ifndef CMAESAG_HPP
#define CMAESAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/functional/hash.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/ARLAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "MLP.hpp"
#include "cmaes_interface.h"

class CMAESAg : public arch::ARLAgent<> {
 public:
  CMAESAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : ARLAgent(_nb_motors), nb_sensors(_nb_sensors){

  }

  virtual ~CMAESAg() {
    cmaes_exit(evo);
    delete evo;
    delete ann;
  }

  const std::vector<double>& _run(double , const std::vector<double>& sensors,
                                 bool learning, bool, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, policy_stochasticity);
        delete next_action;
        next_action = randomized_action;
      } else {
        if(bib::Utils::rand01() < policy_stochasticity)
          for (uint i = 0; i < next_action->size(); i++)
            next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_a         = pt->get<int>("agent.hidden_unit_a");
    lecun_activation      = pt->get<bool>("agent.lecun_activation");
    population            = pt->get<uint>("agent.population");
    gaussian_policy       = pt->get<bool>("agent.gaussian_policy");
    policy_stochasticity  = pt->get<double>("agent.policy_stochasticity");
    
    
    if(hidden_unit_a == 0){
      LOG_ERROR("Linear MLP");
      exit(1);
//       ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    }
     else {
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
      fann_set_training_algorithm(ann->getNeuralNet(), FANN_TRAIN_INCREMENTAL);
    }
    
    struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
    fann_get_connection_array(ann->getNeuralNet(), connections);
    const uint dimension = (nb_sensors+1)*hidden_unit_a + (hidden_unit_a+1)*nb_motors;
    double* startx  = new double[dimension];
    double* deviation  = new double[dimension];
    for(uint j=0; j< dimension; j++){
        startx[j] = connections[j].weight;
        deviation[j] = 0.3;
    }
    free(connections);
    
    evo = new cmaes_t;
    arFunvals = cmaes_init(evo, dimension, startx, deviation, 0, population, NULL/*"config.cmaes.ini"*/);
//     evo->sp.stopTolFun = 1e-150;
//     evo->sp.stopTolFunHist = 1e-150;
//     evo->sp.stopTolUpXFactor = 1e50;
    
    printf("%s\n", cmaes_SayHello(evo));
    new_population();
    LOG_DEBUG(cmaes_Get(evo, "lambda"));
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

  void start_instance(bool learning) override {
    last_action = nullptr;
    scores.clear();

    //put individual into NN
    const double* parameters = nullptr;
    if(learning)
      parameters = pop[current_individual];
    else
      parameters = cmaes_GetPtr(evo, "xbestever");
    const uint dimension = (uint) cmaes_Get(evo, "dim");
    
    struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
    fann_get_connection_array(ann->getNeuralNet(), connections);
    
//     LOG_DEBUG("DIM: " <<dimension << " " << fann_get_total_connections(ann->getNeuralNet()));
    ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
    for(uint j=0; j< dimension; j++)
      connections[j].weight=parameters[j];
    
    fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
    free(connections);
    
    //bib::Logger::PRINT_ELEMENTS(parameters, dimension);
    
    //LOG_FILE("policy_exploration", ann->hash());
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
        new_population();
      }
    }
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

  void save(const std::string& path) override {
    ann->save(path+".actor");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << " " << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

  void _dump(std::ostream& out) const override {
    out << " " << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

 private:
  uint nb_sensors;

  double policy_stochasticity;
  uint hidden_unit_a, population;

  bool lecun_activation, gaussian_policy;

  std::shared_ptr<std::vector<double>> last_action;
  
  std::list<double> scores;

  MLP* ann;
  cmaes_t* evo;
  double *const *pop;
  double *arFunvals;
  uint current_individual;
};

#endif

