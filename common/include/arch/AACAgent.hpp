#ifndef AACAGENT_H
#define AACAGENT_H

#include <vector>
#include <string>

#include "bib/MetropolisHasting.hpp"
#include "AAgent.hpp"

/**
 * @brief architecture
 *
 * The framework is divided into 3 parts : the agent, the environment, and the Simulator
 * that manage her interactions
 */
namespace arch {

enum policy_type {
  GAUSSIAN,
  GREEDY
};

template <typename Pol_Impl>
class Policy{
public:
    Policy(Pol_Impl* _impl, policy_type _type, double _theta):impl(_impl), type(_type), theta(_theta){
      
    }
        
    Policy(Policy& p) : impl(new Pol_Impl(*p.impl)), type(p.type), theta(p.theta){
      
    }
    
    ~Policy(){
      delete impl;
    }
    
    std::vector<double>* run(std::vector<double>& perceptions){
      vector<double>* next_action = impl->computeOut(perceptions);
      
      if(type == policy_type::GAUSSIAN){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, theta);
        delete next_action;
        next_action = randomized_action;
      } else if(type == policy_type::GREEDY && bib::Utils::rand01() < theta){ //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
      
      return next_action;
    }
    
private:
    Pol_Impl* impl;
    policy_type type;
    double theta;
};

template <typename Pol_Impl, typename ProgOptions = AgentProgOptions >
class AACAgent : public AAgent<ProgOptions> {
 public:

  void unique_invoke(boost::property_tree::ptree* inifile,
                     boost::program_options::variables_map* command_args) override {
    _unique_invoke(inifile, command_args);
    if (command_args->count("load"))
      load((*command_args)["load"].as<std::string>());
    
    _gamma                   = inifile->get<double>("agent.gamma");
  }
  
  double getGamma(){
      return _gamma;
  }
  
  virtual double criticEval(const std::vector<double>& perceptions) = 0;
  
  virtual Policy<Pol_Impl>* getCopyCurrentPolicy() = 0;

   protected:
  /**
  * @brief Called only at the creation of the agent.
  * You have to overload this method if you want to get parameters from ini file or from command line.
  *
  * @param inifile
  * @param command_args
  * @return void
  */
  virtual void _unique_invoke(boost::property_tree::ptree* , boost::program_options::variables_map*) {}

  /**
  * @brief To load your previous agent saved to a file.
  * @param filepath the file to load
  *
  * @return void
  */
  virtual void load(const std::string&) {}

private:
  double _gamma;
};
}  // namespace arch

#endif  // AACAGENT_H
