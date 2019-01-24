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
#include "cmaes_interface.h"

double expo_sum(const std::vector<double>& v){
  double sum = 0.f;
  for(double q : v)
    sum += exp(q);
  return sum;
}

class PIDControllerLearn : public arch::ARLAgent<arch::AgentProgOptions> {
 public:
  PIDControllerLearn(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::ARLAgent<arch::AgentProgOptions>(_nb_motors, _nb_sensors), nb_sensors(_nb_sensors), 
    current_param(dimension_problem())
    {

  }

  virtual ~PIDControllerLearn() {
    cmaes_exit(evo);
    delete evo;
    delete ac_informed_state;
  }
  
  void print_final_best(){
    LOG_DEBUG(cmaes_Get(evo, "fbestever"));
    const double* parameters = cmaes_GetPtr(evo, "xbestever");
    bib::Logger::PRINT_ELEMENTS(parameters, dimension_problem());
  }
  
  uint dimension_problem(){
//     return this->nb_motors * 3;
//     return this->nb_motors * 2;
//     return 2;
    return 3;
  }

  const std::vector<double>& _run(double, const std::vector<double>& sensors,
                                  bool, bool goal, bool) override {

    //force a small perturbation
    vector<double>* next_action = new std::vector<double>(this->nb_motors, 0.f);

//     std::vector<double> test = {-2.53168, -0.20791, 0.15605, 0.35428, -0.02659, 0.32720, -1.81081, -0.28895, -0.18864, -2.23505, -0.37164, 0.78502, -3.05356, -0.38279, -0.68477, -0.50092, -0.57394, -0.45777};//best sum max
//     std::vector<double> test = {-1.74053, 0.07579, 0.57802, -0.67090, -0.67155, 0.35731, -2.26236, 0.74043, -0.47473, -2.33031, -0.45227, 0.12848, -1.66925, 1.28023, -0.45736, 0.47006, -0.35110, 0.35893};
    //         std::vector<double> test = {-2.05891, -0.52004, -0.12384};//best old
//         std::vector<double> test = {-4.56065, -0.87200, 0.07312};//best sum max
//     std::vector<double> test = {-2.59888, -0.47748, 0.09128};//best sum expo
    std::vector<double> test = {-2.5, -0.5, 0.1};//best sum expo
    uint k=0;
    for(auto d : test)
      current_param[k++]=d;
    
    uint y=0;
    for(uint i=0; i<this->nb_motors; i++) {
      uint st_index = ac_informed_state->at(i*2);
      if(dimension_problem()== this->nb_motors * 2)
        next_action->at(i) = (2.0f/M_PI) * atan(current_param[y]*sensors[st_index] + current_param[y+1] * sensors[st_index+1]);
      else if(dimension_problem()== this->nb_motors * 3)
        next_action->at(i) = (2.0f/M_PI) * atan(current_param[y]*sensors[st_index] + current_param[y+1] * sensors[st_index+1])*current_param[y+2];
      else if(dimension_problem()==2)
        next_action->at(i) = (2.0f/M_PI) * atan(current_param[0]*sensors[st_index] + current_param[1] * sensors[st_index+1]);
      else if(dimension_problem()==3)
        next_action->at(i) = (2.0f/M_PI) * atan(current_param[0]*sensors[st_index] + current_param[1] * sensors[st_index+1])*current_param[2];
      
      if(dimension_problem()==this->nb_motors * 2)
        y+=2;
      else if(dimension_problem()==this->nb_motors * 3)
        y+=3;
    }

    for(uint i=0; i<ac_informed_state->size(); i++){
      if(i % 2 == 0 && episode_score_max[i/2] < fabs(sensors[ac_informed_state->at(i)])){
        episode_score_max[i/2] = fabs(sensors[ac_informed_state->at(i)]) ;
      }
      
      if(i % 2 == 0) // current angles
        episode_score += fabs(sensors[ac_informed_state->at(i)]);
//       else // derivative less important
//         episode_score += fabs(sensors[ac_informed_state->at(i)])*0.1;
    }
    
//     small action noise
    for (uint i = 0; i < next_action->size(); i++)
      next_action->at(i) += bib::Utils::randin(-0.05f, 0.05f);
    
    if(goal)
      episode_score += 100000;

//  CMA-ES already implement exploration in parameter space
    last_action.reset(next_action);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    ac_informed_state           = bib::to_array<uint>(pt->get<std::string>("devnn.ac_informed_state"));
    population                  = pt->get<uint>("agent.population");
    initial_deviation           = pt->get<double>("agent.initial_deviation");

    check_feasible = true;
    racing = false;
    error_count = 0;

    try {
      check_feasible = pt->get<bool>("agent.check_feasible");
    } catch(boost::exception const& ) {
    }

    try {
      racing = pt->get<bool>("agent.racing");
    } catch(boost::exception const& ) {
    }

    episode = 0;

    uint dimension = dimension_problem();
    double* startx  = new double[dimension];
    double* deviation  = new double[dimension];
    for(uint j=0; j< dimension; j++) {
      deviation[j] = initial_deviation;
      startx[j] = j % 2 == 0 ? -2.f : -0.05f;
    }

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

  bool is_feasible(const double* parameters) {
    for(uint i=0; i < (uint) cmaes_Get(evo, "dim"); i++)
      if(fabs(parameters[i]) >= 50.f) {
        return false;
      }

    return true;
  }

  void new_population() {
    const char * terminate =  cmaes_TestForTermination(evo);
    if(terminate) {
      LOG_INFO("mismatch "<< terminate);
      error_count++;

      if(error_count > 20 && racing) {
        LOG_FILE(DEFAULT_END_FILE, "-1");
        exit(0);
      }
    }
    //ASSERT(!cmaes_TestForTermination(evo), "mismatch "<< cmaes_TestForTermination(evo));

    current_individual = 0;
    pop = cmaes_SamplePopulation(evo);

    if(check_feasible) {
      //check that the population is feasible
      bool allfeasible = true;
      for (int i = 0; i < cmaes_Get(evo, "popsize"); ++i)
        while (!is_feasible(pop[i])) {
          cmaes_ReSampleSingle(evo, i);
          allfeasible = false;
        }

      if(!allfeasible)
        LOG_INFO("non feasible solution produced");
    }
  }

  void _start_episode(const std::vector<double>&, bool) override {
    episode_score = 0;
    
    episode_score_max.clear();
    for(uint i=0; i<ac_informed_state->size(); i++)
      if(i % 2 == 0){
        episode_score_max.push_back(std::numeric_limits<double>::lowest());
      }
  }

  void start_instance(bool learning) override {
    last_action = nullptr;
    scores.clear();

    //put individual into NN
    const double* parameters = nullptr;
    if(learning || !cmaes_UpdateDistribution_done_once)
      parameters = pop[current_individual];
    else
      parameters = cmaes_GetPtr(evo, "xbestever");

    loadPolicyParameters(parameters);

    if(learning)
      episode++;
  }

  void end_episode(bool) override {
    double sum = std::accumulate(episode_score_max.begin(), episode_score_max.end(), 0.f);
//     double sum =expo_sum(episode_score_max);
//     scores.push_back(episode_score);
    scores.push_back(sum);
  }

  void end_instance(bool learning) override {
    if(learning) {
      arFunvals[current_individual] = std::accumulate(scores.begin(), scores.end(), 0.f) / scores.size();

      current_individual++;
      if(current_individual >= cmaes_Get(evo, "lambda")) {
        cmaes_UpdateDistribution(evo, arFunvals);
        cmaes_UpdateDistribution_done_once=true;
        new_population();
      }
    }
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(8) << std::fixed << std::setprecision(5) << episode_score << " " << 
    std::accumulate(episode_score_max.begin(), episode_score_max.end(), 0.f) << " " << expo_sum(episode_score_max);
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(8) << std::fixed << std::setprecision(5) << episode_score << " " <<
    std::accumulate(episode_score_max.begin(), episode_score_max.end(), 0.f) << " "<< expo_sum(episode_score_max); 
  }

 private:
  void loadPolicyParameters(const double* parameters) {
    for(uint i=0; i<current_param.size(); i++)
      current_param[i]=parameters[i];
  }

 private:
  //initilized by constructor
  uint nb_sensors;

  //initialized by invoke
  std::vector<uint>* ac_informed_state = nullptr;
  std::vector<double> current_param;
  uint population;
  double initial_deviation;
  bool check_feasible;
  bool racing;
  cmaes_t* evo;
  double *arFunvals;
  double episode_score;
  std::vector<double> episode_score_max;

  //internal mecanisms
  std::shared_ptr<std::vector<double>> last_action;
  std::list<double> scores;
  double *const *pop;
  bool cmaes_UpdateDistribution_done_once = false;
  uint current_individual;
  uint episode;
  uint error_count;

};

#endif

