#ifndef AACAGENT_H
#define AACAGENT_H

#include <vector>
#include <string>

#include "bib/MetropolisHasting.hpp"
#include "ARLAgent.hpp"

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
class AACAgent : public ARLAgent<ProgOptions> {
 public:
  AACAgent(uint _nb_motors):ARLAgent<ProgOptions>(_nb_motors){
    
  }

  double getGamma(){
      return this->gamma;
  }
  
  virtual double criticEval(const std::vector<double>& perceptions) = 0;
  
  virtual Policy<Pol_Impl>* getCopyCurrentPolicy() = 0;
};
}  // namespace arch

#endif  // AACAGENT_H
