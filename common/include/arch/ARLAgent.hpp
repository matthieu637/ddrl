
#ifndef ARLAGENT_H
#define ARLAGENT_H

#include <vector>
#include <string>

#include "arch/Dummy.hpp"
#include "arch/AAgent.hpp"
// TODO:finished with action history
/**
 * @brief architecture
 *
 * The framework is divided into 3 parts : the agent, the environment, and the Simulator
 * that manage her interactions
 */
namespace arch {

template <typename ProgOptions = AgentProgOptions>
class ARLAgent : public AAgent<ProgOptions> {
 public:
  ARLAgent(uint _nb_motors, uint _nb_sensors):
  returned_ac(_nb_motors), nb_sensors(_nb_sensors), nb_motors(_nb_motors){
    
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
    const std::vector<double> *state_to_send = &perceptions;
    if(history_size > 1){
      for (uint h = history_size - 1 ; h >= 1; h--)
        std::copy(returned_st.begin() + (h-1) * nb_sensors,
                  returned_st.begin() + h * nb_sensors, 
                  returned_st.begin() + h * nb_sensors);
      
      std::copy(perceptions.begin(), perceptions.end(), returned_st.begin());
      state_to_send = &returned_st;
    }
    inter_rewards.push_back(r);
    time_for_ac--;
    if (time_for_ac == 0 || absorbing_state || finished) {
      double reward = *std::max_element(inter_rewards.begin(), inter_rewards.end());
      
      //take the last reward if go to absorbing state
      //because the last will be lower than the others before
      if(absorbing_state){
        reward = r;
      }
 
      _last_receive_reward = reward;
      const std::vector<double>& next_action = _run(reward*reward_scale, *state_to_send, learning, absorbing_state, finished);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      inter_rewards.clear();
      sum_weighted_reward += reward * global_pow_gamma;
      sum_reward += reward;
      global_pow_gamma *= gamma;
    }

    return returned_ac;
  }
  
  double last_receive_reward(){
      return _last_receive_reward;
  }

  bool did_decision() {
      return time_for_ac == decision_each;
  }
  
  uint get_decision_each(){
      return decision_each;
  }
  
  double sum_weighted_rewards(){
      return sum_weighted_reward;
  }
  
  uint get_number_motors(){
      return nb_motors;
  }
  
  uint get_number_sensors(){
      return nb_sensors;
  }
  
  uint get_state_size(){
      return state_size;
  }

  /**
   * @brief This method is called after each beginning of a new instance of episode
   * @param sensors the first perceptions from the environment
   * @return void
   */
  void start_episode(const std::vector<double>& perceptions, bool learning) override {
    time_for_ac = 1;
    inter_rewards.clear();
    if(last_episode_learning)
      last_sum_weighted_reward = sum_weighted_reward;
    sum_weighted_reward = 0;
    sum_reward = 0;
    last_episode_learning = learning;
    
    global_pow_gamma = 1.000000000f;
    
    if(history_size == 1)
      _start_episode(perceptions, learning);
    else {
      if(get_state_size()==0){
        LOG_ERROR("fix your agent constructor " << nb_sensors <<  " " << history_size );
        exit(1);
      }
      returned_st.resize(get_state_size());
      for (uint h = 0 ; h < history_size; h++)
        std::copy(perceptions.begin(), perceptions.end(), returned_st.begin() + h * nb_sensors);
      if(action_in_history){
        for (uint h = 0 ; h < history_size; h++)
          for (uint j = 0 ; j < nb_motors; j++)
            returned_st[history_size*nb_sensors + h * nb_motors + j] = 0.f;
      }
      _start_episode(returned_st, learning);
    }
  }
  
  void unique_invoke(boost::property_tree::ptree* inifile,
                     boost::program_options::variables_map* command_args,
                     bool forbidden_load) override {
    gamma                 = inifile->get<double>("agent.gamma");
    decision_each         = inifile->get<int>("agent.decision_each");
    history_size          = 1;
    action_in_history     = false;
    reward_scale          = 1.0f;
    try {
      history_size        = inifile->get<int>("agent.history_size");
    } catch(boost::exception const& ) {
    }
    try {
      action_in_history   = inifile->get<bool>("agent.action_in_history");
    } catch(boost::exception const& ) {
    }
    
    try {
      reward_scale        = inifile->get<double>("agent.reward_scale");
    } catch(boost::exception const& ) {
    }
    
    if(history_size == 0){
      LOG_ERROR("cannot have 0 history");
      exit(1);
    }
    
    try {
      if(inifile->get<int>("environment.max_step_per_instance") % decision_each != 0){
          LOG_ERROR("please synchronize environment.max_step_per_instance with agent.decision_each");
          exit(1);
      }
    } catch(boost::exception const& ) {
    }
    state_size = nb_sensors * history_size;
    //+ (action_in_history ? nb_motors * history_size : 0 );
    
    _unique_invoke(inifile, command_args);
    if (command_args->count("load") && !forbidden_load)
      this->load((*command_args)["load"].as<std::string>());
  }
  
  
  void provide_early_development(AAgent<ProgOptions>* _old_ag) override {
    old_ag = _old_ag;
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
  virtual void _unique_invoke(boost::property_tree::ptree* , boost::program_options::variables_map*) override {}


private:
  std::list<double> inter_rewards;
  double global_pow_gamma;
  uint time_for_ac;
  
  std::vector<double> returned_ac;
  double _last_receive_reward;
  std::vector<double> returned_st;
  uint nb_sensors;
  double reward_scale;
  
protected:
  double sum_weighted_reward = 0;
  double sum_reward = 0;
  double last_sum_weighted_reward = 0;
  bool last_episode_learning = true;
  double gamma;
  uint nb_motors;
  uint state_size;
  uint decision_each;
  uint history_size;
  bool action_in_history;
  AAgent<ProgOptions>* old_ag;
};
}  // namespace arch

#endif  // AAGENT_H
