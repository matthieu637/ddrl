#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <string>
#include <vector>
#include <list>
#include <type_traits>
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/ini_parser.hpp"
#include "boost/program_options.hpp"

#include "bib/Assert.hpp"
#include "bib/Logger.hpp"
#include "bib/Utils.hpp"
#include "bib/Chrono.hpp"
#include "bib/Dumper.hpp"
#include "arch/AAgent.hpp"
#include "arch/AEnvironment.hpp"
#include "arch/DefaultParam.hpp"

using std::string;

namespace arch {

template <typename Environment, typename Agent>
class Simulator {
  //     static_assert(std::is_base_of<AEnvironment<>, Environment>::value,
  //     "Environment should be a base of AEnvironment.");
  //     static_assert(std::is_base_of<AAgent<>, Agent>::value, "Agent should be
  //     a base of AAgent.");

 public:
  Simulator() : max_episode(0), test_episode_per_episode(0), test_episode_at_end(0),
    dump_log_each(0), display_log_each(0), save_agent_each(0), properties(nullptr),
    command_args(nullptr), time_spend(), env(nullptr), agent(nullptr) {}

  ~Simulator() {
    delete properties;
    delete command_args;
  }

  void init(int argc, char** argv) {
    string config_file = DEFAULT_CONFIG_FILE;
    readCommandArgs(argc, argv, &config_file);
    readConfig(config_file);
  }

  void run() {
    ASSERT(well_init, "Please call init() first on Simulator");

    env = new Environment;
    env->unique_invoke(properties, command_args);

    agent = new Agent(env->number_of_actuators(), env->number_of_sensors());
    agent->unique_invoke(properties, command_args);

    time_spend.start();
    for (unsigned int episode = 0; episode < max_episode; episode++) {
      //  learning
      run_episode(true, episode);

      for (unsigned int test_episode = 0; test_episode < test_episode_per_episode; test_episode++) {
        //  testing during learning
        run_episode(false, episode);
      }
    }

    for (unsigned int test_episode = 0; test_episode < test_episode_at_end;
         test_episode++) {
      //  testing after learning
      run_episode(false, test_episode);
    }

    env->unique_destroy();
    delete env;

    LOG_FILE(DEFAULT_END_FILE,
             "" << (float)(time_spend.finish() / 60.f));  // in minutes
  }

 private:
  void run_episode(bool learning, unsigned int episode) {
    env->reset_episode();
    std::list<float> all_rewards;

    while (env->hasInstance()) {
      std::vector<float> perceptions = env->perceptions();
      agent->start_episode(perceptions);

      while (env->running()) {
        perceptions = env->perceptions();
        float reward = env->performance();
        const std::vector<float>& actuators =
          agent->run(reward, perceptions, learning, false);
        env->apply(actuators);
        all_rewards.push_back(reward);
      }

      // if the environment is in a final state
      //        i.e it didn't reach the number of step but finished well
      // then we call the algorithm a last time to give him this information
      if (env->final_state()) {
        perceptions = env->perceptions();
        float reward = env->performance();
        agent->run(reward, perceptions, learning, true);
        all_rewards.push_back(reward);
      }

      env->next_instance();
      agent->end_episode();
    }

    dump_and_display(episode, all_rewards, env, agent, learning);
    save_agent(agent, episode);
  }

  void dump_and_display(unsigned int episode, const std::list<float>& all_rewards, Environment* env,
                        Agent* ag, bool learning) {
    bool display = episode % display_log_each == 0;
    bool dump = episode % dump_log_each == 0;

    if (dump || display) {
      bib::Utils::V3M reward_stats = bib::Utils::statistics(all_rewards);

      if (display) {
        bib::Dumper<Environment, bool, bool> env_dump(env, true, false);
        bib::Dumper<Agent, bool, bool> agent_dump(ag, true, false);
        LOG_INFO((learning ? "L " : "T ")
                 << std::left << std::setw(7) << std::setfill(' ') << episode
                 << std::left << std::setw(8) << std::setfill(' ') << std::setprecision(4) << reward_stats.mean
                 << std::left << std::setw(8) << std::setfill(' ') << std::setprecision(4) << reward_stats.var
                 << std::left << std::setw(8) << std::setfill(' ') << std::setprecision(4) << reward_stats.max
                 << std::left << std::setw(8) << std::setfill(' ') << std::setprecision(4) << reward_stats.min
                 << " " << env_dump << " " << agent_dump);
      }

      if (dump) {
        bib::Dumper<Environment, bool, bool> env_dump(env, false, true);
        bib::Dumper<Agent, bool, bool> agent_dump(ag, false, true);
        LOG_FILE(learning ? DEFAULT_DUMP_LEARNING_FILE : DEFAULT_DUMP_TESTING_FILE,
                 episode << " " << reward_stats.mean << " " << reward_stats.var << " " <<
                 reward_stats.max << " " << reward_stats.min << env_dump << agent_dump);
      }
    }
  }

  void save_agent(Agent* agent, unsigned int episode) {
    if (episode % save_agent_each == 0 && episode != 0) {
      std::string filename(DEFAULT_AGENT_SAVE_FILE);
      std::string filename2 = std::to_string(episode);
      std::string path = filename + filename2;
      agent->save(path);
    }
  }

  void readCommandArgs(int argc, char** argv, string* s) {
    namespace po = boost::program_options;

    po::options_description desc("Allowed Simulator options");
    desc.add(Environment::program_options());
    desc.add(Agent::program_options());
    desc.add_options()
    ("config", po::value<string>(), "set the config file to load [default : config.ini]")
    ("help", "produce help message");

    command_args = new po::variables_map;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, *command_args);
    po::notify(*command_args);

    if (command_args->count("help")) {
      std::cout << "Usage : Simulator [options]" << std::endl;
      std::cout << desc;
      exit(0);
    }

    if (command_args->count("config")) {
      *s = (*command_args)["config"].as<string>();
    }
  }

  void readConfig(string config_file = DEFAULT_CONFIG_FILE) {
    properties = new boost::property_tree::ptree;
    boost::property_tree::ini_parser::read_ini(config_file, *properties);

    max_episode                 = properties->get<unsigned int>("simulation.max_episode");
    test_episode_per_episode    = properties->get<unsigned int>("simulation.test_episode_per_episode");
    test_episode_at_end         = properties->get<unsigned int>("simulation.test_episode_at_end");

    dump_log_each               = properties->get<unsigned int>("simulation.dump_log_each");
    display_log_each            = properties->get<unsigned int>("simulation.display_log_each");
    save_agent_each             = properties->get<unsigned int>("simulation.save_agent_each");

#ifndef NDEBUG
    well_init = true;
#endif
  }

 private:
  unsigned int max_episode;
  unsigned int test_episode_per_episode;
  unsigned int test_episode_at_end;

  unsigned int dump_log_each;
  unsigned int display_log_each;
  unsigned int save_agent_each;

  boost::property_tree::ptree* properties;
  boost::program_options::variables_map* command_args;

  bib::Chrono time_spend;

  Environment* env;
  Agent* agent;

#ifndef NDEBUG
  bool well_init = false;
#endif
};
}  // namespace arch

#endif
