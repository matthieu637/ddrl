
#ifndef ARLAGENT_H
#define ARLAGENT_H

#include <vector>
#include <string>

#include "arch/Dummy.hpp"
#include "arch/AAgent.hpp"

/**
 * @brief architecture
 *
 * The framework is divided into 3 parts : the agent, the environment, and the Simulator
 * that manage her interactions
 */
namespace arch {

template <typename ProgOptions = AgentProgOptions>
class ARLAgent : public AAgent<AgentProgOptions> {
 public:
  ARLAgent(uint _nb_motors):returned_ac(_nb_motors), nb_motors(_nb_motors){
    
  }
   
  /**
   * @brief This is the main method to define the behavior of your agent
   *
   * @param reward the reward you got for your last action choose
   * @param perceptions the current perceptions provided by the environment
   * @param learning should the agent learns during the interaction (false to test an agent)
   * @param absorbing_state did the agent reached a global goal during his last action
   * @param finished is it the last step of this episode
   * @return const std::vector< double, std::allocator< void > >&
   */
  const std::vector<double>& runf(double r, const std::vector<double>& perceptions,
                                         bool learning, bool absorbing_state, bool finished) override {
    inter_rewards.push_back(r);
    time_for_ac--;
    if (time_for_ac == 0 || absorbing_state || finished) {
      double reward = *std::max_element(inter_rewards.begin(), inter_rewards.end());
      
      //take the last reward if go to absorbing state
      //because the last will be lower than the others before
      if(absorbing_state){
        reward = r;
      }
        
      const std::vector<double>& next_action = _run(reward, perceptions, learning, absorbing_state, finished);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      inter_rewards.clear();
      sum_weighted_reward += reward * global_pow_gamma;
      global_pow_gamma *= gamma;
    }

    return returned_ac;
  }

  /**
   * @brief This method is called after each beginning of a new instance of episode
   * @param sensors the first perceptions from the environment
   * @return void
   */
  void start_episode(const std::vector<double>& perceptions, bool learning) override {
    time_for_ac = 1;
    inter_rewards.clear();
    sum_weighted_reward = 0;
    global_pow_gamma = 1.000000000f;
    
    _start_episode(perceptions, learning);
  }
  
  void unique_invoke(boost::property_tree::ptree* inifile,
                     boost::program_options::variables_map* command_args) override {
    gamma                 = inifile->get<double>("agent.gamma");
    decision_each         = inifile->get<int>("agent.decision_each");
    
    _unique_invoke(inifile, command_args);
    if (command_args->count("load"))
      load((*command_args)["load"].as<std::string>());
  }
  
protected:
    /**
   * @brief This is the main method to define the behavior of your agent
   *
   * @param reward the reward you got for your last action choose
   * @param perceptions the current perceptions provided by the environment
   * @param learning should the agent learns during the interaction (false to test an agent)
   * @param goal_reached did the agent reached a global goal during his last action
   * @return const std::vector< double, std::allocator< void > >&
   */
  virtual const std::vector<double>& _run(double reward, const std::vector<double>& perceptions,
                                        bool learning, bool goal_reached, bool finished) = 0;
                                        
  virtual void _start_episode(const std::vector<double>& perceptions, bool learning){
    (void) perceptions;
    (void) learning;
  }
  
  /**
  * @brief Called only at the creation of the agent.
  * You have to overload this method if you want to get parameters from ini file or from command line.
  *
  * @param inifile
  * @param command_args
  * @return void
  */
  virtual void _unique_invoke(boost::property_tree::ptree* , boost::program_options::variables_map*) {}


private:
  std::list<double> inter_rewards;
  double global_pow_gamma;
  uint time_for_ac;
  uint decision_each;
  
  std::vector<double> returned_ac;
  
protected:
  double sum_weighted_reward;
  double gamma;
  uint nb_motors;
};
}  // namespace arch

#endif  // AAGENT_H
