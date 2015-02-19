
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"

class Test {
public:
    static boost::program_options::options_description program_options;

};

boost::program_options::options_description Test::program_options;


int main(int argc, char **argv)
{
    arch::Simulator<Test,Test> s;
    s.init(argc, argv);
    
    s.run();
    
    LOG_DEBUG("works !");
}
