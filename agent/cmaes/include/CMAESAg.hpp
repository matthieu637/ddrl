#ifndef CMAESAG_HPP
#define CMAESAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/AAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "MLP.hpp"
#include "cmaes_interface.h"

class CMAESAg : public arch::AAgent<> {
 public:
  CMAESAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~CMAESAg() {
    cmaes_exit(evo);
    delete evo;
    delete ann;
  }

  const std::vector<double>& run(double r, const std::vector<double>& sensors,
                                bool learning, bool goal_reached) override {

    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<double>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }

  const std::vector<double>& _run(double , const std::vector<double>& sensors,
                                 bool, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

//     if(learning) {
//       if(gaussian_policy){
//         vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
//         delete next_action;
//         next_action = randomized_action;
//       } else {
//         if(bib::Utils::rand01() < noise)
//           for (uint i = 0; i < next_action->size(); i++)
//             next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
//       }
//     }
    last_action.reset(next_action);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    gamma               = pt->get<double>("agent.gamma");
    hidden_unit_a       = pt->get<int>("agent.hidden_unit_a");
    decision_each       = pt->get<int>("agent.decision_each");
    lecun_activation    = pt->get<bool>("agent.lecun_activation");
    population          = pt->get<uint>("agent.population");
    
    
    if(hidden_unit_a == 0){
      LOG_ERROR("Linear MLP");
      exit(1);
//       ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    }
     else {
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
      fann_set_training_algorithm(ann->getNeuralNet(), FANN_TRAIN_INCREMENTAL);
    }
    
    evo = new cmaes_t;
    arFunvals = cmaes_init(evo, (nb_sensors+1)*hidden_unit_a + (hidden_unit_a+1)*nb_motors, NULL, NULL, 0, 30, "config.cmaes.ini");
    printf("%s\n", cmaes_SayHello(evo));
    new_population();
    LOG_DEBUG(cmaes_Get(evo, "lambda"));
  }
  
  void new_population(){
    ASSERT(!cmaes_TestForTermination(evo), "mismatch");
    
    current_individual = 0;
    pop = cmaes_SamplePopulation(evo);
  }

  void start_episode(const std::vector<double>&) override {
    last_action = nullptr;

    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;

    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
    
    //put individual into NN
    const double* parameters = pop[current_individual];
    const uint dimension = (uint) cmaes_Get(evo, "dim");
    
    struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
    fann_get_connection_array(ann->getNeuralNet(), connections);
    
//     LOG_DEBUG("DIM: " <<dimension << " " << fann_get_total_connections(ann->getNeuralNet()));
    ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
    for(uint j=0; j< dimension; j++){
//       LOG_DEBUG("##"<< j <<"##");
//       LOG_DEBUG(connections[j].weight);
//       LOG_DEBUG(parameters[j]);
      connections[j].weight=parameters[j];
    }
    
    fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
    free(connections);
  }

  void end_episode() override {
    arFunvals[current_individual] = -sum_weighted_reward;
    
    current_individual++;
    if(current_individual >= cmaes_Get(evo, "lambda")){
      cmaes_UpdateDistribution(evo, arFunvals);
      new_population();
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
    out << sum_weighted_reward << " " << std::setw(8) << std::fixed << std::setprecision(5);
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5);
  }

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;

  uint internal_time, decision_each;

  double gamma;
  uint hidden_unit_a, population;

  bool lecun_activation;

  std::shared_ptr<std::vector<double>> last_action;

  std::vector<double> returned_ac;

  MLP* ann;
  cmaes_t* evo;
  double *const *pop;
  double *arFunvals;
  uint current_individual;
};

#endif

