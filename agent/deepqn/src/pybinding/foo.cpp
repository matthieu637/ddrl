#include "arch/Simulator.hpp"
#include "FusionOOAg.hpp"

extern "C" {
  FusionOOAg* FOOAg_new(uint a, uint b) {
    FLAGS_minloglevel = 2;
    google::InitGoogleLogging("");
    google::InstallFailureSignalHandler();
    return new FusionOOAg(a,b);
  }

  void FOOAg_unique_invoke(FusionOOAg* ag, int argc, char** argv) {
    namespace po = boost::program_options;
    boost::property_tree::ptree* properties;
    boost::program_options::variables_map* command_args;

    string config_file = DEFAULT_CONFIG_FILE;

    po::options_description desc("Allowed Simulator options");
    desc.add(FusionOOAg::program_options());
    desc.add_options()
    ("config", po::value<std::vector<string>>(), "set the config file to load [default : config.ini]")
    ("help", "produce help message");

    command_args = new po::variables_map;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    po::store(parsed, *command_args);
    po::notify(*command_args);

    if (command_args->count("config")) {
      config_file = (*command_args)["config"].as<std::vector<string>>()[0];
    }

    properties = new boost::property_tree::ptree;
    boost::property_tree::ini_parser::read_ini(config_file, *properties);
    ag->unique_invoke(properties, command_args, false);

    delete properties;
    delete command_args;
  }

  void FOOAg_start_episode(FusionOOAg* ag, const double* _state, bool learning) {
    std::vector<double> state(_state, _state+ag->get_number_sensors());
    return ag->start_episode(state, learning);
  }

  void FOOAg_end_episode(FusionOOAg* ag, bool learning) {
    ag->end_episode(learning);
    ag->end_instance(learning);
  }

  const double* FOOAg_run(FusionOOAg* ag, double reward, const double* sensors,
                              bool learning, bool goal_reached, bool last) {
    std::vector<double> state(sensors, sensors+ag->get_number_sensors());
    const std::vector<double>& ac = ag->runf(reward, state, learning, goal_reached, last);
    return ac.data();
  }

  void FOOAg_dump(FusionOOAg* ag, bool learning, int episode, int step, double treward) {
    bib::Dumper<FusionOOAg, bool, bool> agent_dump(ag, false, true);
    LOG_FILE(std::to_string(0) + (learning ? DEFAULT_DUMP_LEARNING_FILE : DEFAULT_DUMP_TESTING_FILE),
             episode << " " << step << " " << treward << " " << agent_dump);
  }

  void FOOAg_display(FusionOOAg* ag, bool learning, int episode, int step, double treward) {
    bib::Dumper<FusionOOAg, bool, bool> agent_dump(ag, true, false);
    LOG_INFO((learning ? "L " : "T ")
             << std::left << std::setw(6) << std::setfill(' ') << episode
             << std::left << std::setw(7) << std::fixed << step
             << std::left << std::setw(7) << std::fixed << treward
             << " " << agent_dump);
  }
  
  void FOOAg_save(FusionOOAg* ag, int episode) {
    std::string path = "agent" + std::to_string(episode);
    ag->save(path, true, true);
  }
  
  void FOOAg_load(FusionOOAg* ag, int episode) {
    std::string path = "agent" + std::to_string(episode);
    ag->load(path);
  }
}
