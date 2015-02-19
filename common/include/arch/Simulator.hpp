#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "bib/Assert.hpp"

#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>
#include "bib/Logger.hpp"
#include <bib/Chrono.hpp>

#define DEFAULT_CONFIG_FILE "config.ini"
#define DEFAULT_END_FILE "time_elapsed"

using std::string;

namespace arch
{
template <typename Environment, typename Agent>
class Simulator
{
public:
    Simulator() {}

    ~Simulator() {
        delete properties;
        delete command_args;
    }

    void init(int argc, char **argv) {
        string config_file = DEFAULT_CONFIG_FILE;
        readCommandArgs(argc, argv, &config_file);
        readConfig(config_file);
    }

    void run() {
        ASSERT(well_init, "Please call init() first on Simulator");

        env = new Environment;
        env->unique_invoke(properties);
	
	agent = new Agent;
	agent->unique_invoke(properties);

        time_spend.start();
        for(unsigned int episode=0; episode < max_episode; episode++) {
//	learning
            run_episode(true);

            for(unsigned int test_episode=0; test_episode < test_episode_per_episode ; test_episode++) {
//	    testing during learning
                run_episode(true);
            }
        }

        for(unsigned int test_episode=0; test_episode < test_episode_at_end ; test_episode++) {
// 	testing after learning
            run_episode(true);
        }
        
        env->unique_destroy();
	delete env;

        LOG_FILE(DEFAULT_END_FILE, ""<<(float) (time_spend.finish() / 60.f)); //in minutes
    }

private:

    void run_episode(bool learning) {
	env->reset_episode();
	std::vector<float> perceptions = env->perceptions();
 	agent->start_episode(perceptions);
	
	while(env->running()){
	    perceptions = env->perceptions();
	    const std::vector<float>& actuators = agent->run(perceptions, learning);
	    env->apply(actuators);
	}
    }

    void readCommandArgs(int argc, char **argv, string* s) {
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

        if(command_args->count("help")) {
            std::cout << "Usage : Simulator [options]" << std::endl;
            std::cout << desc;
            exit(0);
        }

        if(command_args->count("config")) {
            *s = (*command_args)["config"].as<string>();
        }
    }

    void readConfig(string config_file=DEFAULT_CONFIG_FILE) {
        properties = new  boost::property_tree::ptree;
        boost::property_tree::ini_parser::read_ini(config_file, *properties);
        max_episode = properties->get<unsigned int>("simulation.max_episode");
        test_episode_per_episode = properties->get<unsigned int>("simulation.test_episode_per_episode");
        test_episode_at_end = properties->get<unsigned int>("simulation.test_episode_at_end");
        dump_log_each = properties->get<unsigned int>("simulation.dump_log_each");
        display_log_each = properties->get<unsigned int>("simulation.display_log_each");

#ifndef NDEBUG
        well_init=true;
#endif
    }

private:
    unsigned int max_episode;
    unsigned int test_episode_per_episode;
    unsigned int test_episode_at_end;
    unsigned int dump_log_each;
    unsigned int display_log_each;

    boost::property_tree::ptree* properties;
    boost::program_options::variables_map* command_args;

    bib::Chrono time_spend;
    
    Environment* env;
    Agent* agent;

#ifndef NDEBUG
    bool well_init=false;
#endif
};


}

#endif
