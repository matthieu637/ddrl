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
    Policy(Pol_Impl* _impl, policy_type _type, double _theta, uint _decision_each):impl(_impl), type(_type), theta(_theta), decision_each(_decision_each), time_for_ac(1){
      
    }
        
    Policy(Policy& p) : impl(new Pol_Impl(*p.impl)), type(p.type), theta(p.theta), decision_each(p.decision_each), time_for_ac(1){
      
    }
    
    ~Policy(){
      delete impl;
    }
    
    std::vector<double>& run_td(std::vector<double>& perceptions){
      time_for_ac--;
      if (time_for_ac == 0){
          const std::vector<double>* next_action = run(perceptions);
          time_for_ac = decision_each;
          
          returned_ac.resize(next_action->size());
          for (uint i = 0; i < next_action->size(); i++)
            returned_ac[i] = next_action->at(i);
          delete next_action;
      }
      
      return returned_ac;
    }
    
    bool did_decision() {
      return time_for_ac == decision_each;
    }
    
    void reset_decision_interval(){
      time_for_ac = 1;
    }
    
    bool last_step_before_decision(){
      return time_for_ac == 1;
    }
    
    void disable_stochasticity(){
       type = policy_type::GREEDY;
       theta = 0.00000000000000000f;
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
    uint decision_each;
    
    uint time_for_ac;
    std::vector<double> returned_ac;
};

template <typename Pol_Impl, typename ProgOptions = AgentProgOptions >
class AACAgent : public ARLAgent<ProgOptions> {
 public:
  AACAgent(uint _nb_motors, uint _nb_sensors):ARLAgent<ProgOptions>(_nb_motors, _nb_sensors){
    
  }

  double getGamma(){
      return this->gamma;
  }
  
  virtual double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) = 0;
  
  virtual Policy<Pol_Impl>* getCopyCurrentPolicy() = 0;
  
  virtual void learn_V(std::map<std::vector<double>, double>&){}
};
}  // namespace arch

#endif  // AACAGENT_H
