
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"

class ExampleEnv {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc;
        return desc;
    }

    void unique_invoke(boost::property_tree::ptree* properties) {
        max_step = properties->get<unsigned int>("environment.max_step");
    }

    void unique_destroy() {
    }

    const std::vector<float>& perceptions() const
    {
        return internal_state;
    }

    void reset_episode() {
        current_step = 0;
    }

    void apply(const std::vector<float>&) {
        current_step++;
    }

    bool running() {
        return current_step < max_step;
    }

    unsigned int current_step=0;
    unsigned int max_step;
    std::vector<float> internal_state;
};

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

    const std::vector<float>& run(const std::vector<float>&, bool) {
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
