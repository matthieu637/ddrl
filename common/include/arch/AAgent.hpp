#ifndef AAGENT_H
#define AAGENT_H

#include <vector>
#include <string>

#include "arch/Dummy.hpp"
#include "arch/CommonAE.hpp"

/**
 * @brief architecture
 *
 * The framework is divided into 3 parts : the agent, the environment, and the Simulator
 * that manage her interactions
 */
namespace arch {

/**
 * @brief Abstract Agent class
 * All agent should inherited this class to be run with the Simulator
 *
 * The template parameter serves to add additionnal command line options.
 * @see AgentProgOptions for an example
 *
 * Each agent must define a constructor of the type (unsigned int, unsigned int)
 * the first parameter is the number of motors/actuators provided by the environment
 * and the second is the number of sensors/perceptions.
 */
template <typename ProgOptions = AgentProgOptions>
class AAgent : public ProgOptions, public CommonAE {
 public:
  //     virtual AAgent(unsigned int, unsigned int)=0;

  /**
   * @brief This is the main method to define the behavior of your agent
   *
   * @param reward the reward you got for your last action choose
   * @param perceptions the current perceptions provided by the environment
   * @param learning should the agent learns during the interaction (false to test an agent)
   * @param finished did the agent reached a global goal during his last action
   * @return const std::vector< float, std::allocator< void > >&
   */
  virtual const std::vector<float>& run(float reward, const std::vector<float>& perceptions,
                                        bool learning, bool finished) = 0;

  /**
   * @brief This method is called after each beginning of a new instance of episode
   * @param sensors the first perceptions from the environment
   * @return void
   */
  virtual void start_episode(const std::vector<float>&) {}

  /**
   * @brief This method is called after each end of an instance
   * @return void
   */
  virtual void end_episode() {}

  /**
   * @brief To save your agent to a file.
   * Becareful to save every data if you intend to load your agent from this file later.
   * @param filepath where to save the agent
   *
   * @return void
   */
  virtual void save(const std::string&) {}

  /**
  * @brief To load your previous agent saved to a file.
  * @param filepath the file to load
  *
  * @return void
  */
  virtual void load(const std::string&) {}


  void unique_invoke(boost::property_tree::ptree* inifile,
                     boost::program_options::variables_map* command_args) override {
    _unique_invoke(inifile, command_args);
    if (command_args->count("load"))
      load((*command_args)["load"].as<std::string>());
  }

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
   * @brief if you want to display specific statistics of your agent
   * @param stdout use the operator <<
   * @return void
   */
  void _display(std::ostream&) const override {}

  /**
   * @brief if you want to dump to file specific statistics of your agent
   * @param filestream use the operator <<
   * @return void
   */
  void _dump(std::ostream&) const override {}
};
}  // namespace arch

#endif  // AAGENT_H
