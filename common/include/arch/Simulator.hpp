#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "bib/Assert.hpp"
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/program_options.hpp>

#define DEFAULT_CONFIG_FILE "config.ini"

using std::string;

namespace arch
{
template <typename Environment, typename Agent>
class Simulator
{
public:
    Simulator() {}

    void init(int argc, char **argv) {
        string config_file = DEFAULT_CONFIG_FILE;
        readCommandArgs(argc, argv, &config_file);
        readConfig(config_file);
    }

    void run() {
        ASSERT(well_init, "Please call init() first on Simulator");
        //init singleton
//          bib::Seed::getInstance();

//       	using namespace std::chrono;
// 	high_resolution_clock::time_point begin = high_resolution_clock::now();

//         for(unsigned int episode=0; episode < max_episode; episode++)

    }


private:

    void readCommandArgs(int argc, char **argv, string* s) {
        namespace po = boost::program_options;

        po::options_description desc("Allowed Simulator options");
        desc.add(Environment::program_options);
        desc.add(Agent::program_options);
        desc.add_options()
        ("config", po::value<string>(), "set the config file to load [default : config.ini]")
        ("help", "produce help message");

        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        po::notify(vm);

        if(vm.count("help")) {
            std::cout << "Usage : Simulator [options]" << std::endl;
            std::cout << desc;
            exit(0);
        }

        if(vm.count("config")) {
            *s = vm["config"].as<string>();
        }
    }

    void readConfig(string config_file=DEFAULT_CONFIG_FILE) {
        properties = new  boost::property_tree::ptree;
        boost::property_tree::ini_parser::read_ini(config_file, *properties);
        max_episode = properties->get<unsigned int>("simulation.max_episode");
    }

private:
    unsigned int max_episode;
    boost::property_tree::ptree* properties;

#ifndef NDEBUG
    bool well_init=false;
#endif
};


}

#endif
