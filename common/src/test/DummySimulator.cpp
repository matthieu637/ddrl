
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"

class ExampleEnv {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc;
        return desc;
    }

    void unique_invoke(boost::property_tree::ptree* properties) {
        instance_per_episode = properties->get<unsigned int>("environment.instance_per_episode");
        max_step_per_instance = properties->get<unsigned int>("environment.max_step_per_instance");
    }

    void unique_destroy() {
    }

    const std::vector<float>& perceptions() const
    {
        return internal_state;
    }

    void reset_episode() {
        current_instance = 0;
        current_step = 0;
    }

    void next_instance() {
        current_step = 0;
        current_instance++;
    }

    void apply(const std::vector<float>&) {
        current_step++;
    }

    float performance() {
        return 0;
    }

    bool running() const {
        return current_step < max_step_per_instance;
    }

    bool hasInstance() const {
        return current_instance < instance_per_episode;
    }

    std::ostream& dump_file(std::ostream& out) {
        return out;
    }

    typedef struct {
      ExampleEnv* env;
      bool test;
    } SE;
    
    friend std::ostream& operator<<(std::ostream& out, ExampleEnv::SE &env);

//     friend std::ostream operator

    unsigned int current_step=0;
    unsigned int current_instance=0;

    unsigned int max_step_per_instance;
    unsigned int instance_per_episode;
    std::vector<float> internal_state;
};



std::ostream& operator<< (std::ostream &out, ExampleEnv::SE &env)
{
      return out;
}

class ExampleAgent {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc("Allowed Agent options");
        desc.add_options()
        ("load", boost::program_options::value<std::string>(), "set the agent to load");
        return desc;
    }

    void unique_invoke(boost::property_tree::ptree*) {
    }

    void start_episode(const std::vector<float>&) {

    }

    const std::vector<float>& run(float reward, const std::vector<float>&, bool, bool) {
        return {0};
    }
};


int main(int argc, char **argv)
{
    arch::Simulator<ExampleEnv, ExampleAgent> s;
    s.init(argc, argv);

    s.run();

    LOG_DEBUG("works !");
}
