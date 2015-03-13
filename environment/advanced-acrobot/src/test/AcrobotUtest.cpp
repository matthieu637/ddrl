#include "gtest/gtest.h"

#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

//   arch::Simulator<AdvancedAcrobotEnv, arch::ExampleAgent> s;
//
//   int argc=0;
//   char nul[0] = {};
//   char** argv = &nul;
//   s.init(argc, argv);

//   s.run();

TEST(AdvancedAcrobotEnv, Realisable)
{
    boost::property_tree::ptree properties;
    boost::property_tree::ini_parser::read_ini("utest.ini", properties);
    
    boost::program_options::variables_map command_args;
  
    AdvancedAcrobotEnv env;
    env.unique_invoke(&properties, &command_args);
    arch::ExampleAgent ag(env.number_of_actuators(), env.number_of_sensors());
    std::vector<float> s(env.number_of_sensors(), 0.);

    for (uint i = 0; i < 5000; i++) {
        env.apply(ag.run(0, s, false, false));
        EXPECT_GE(env.performance(), 0.f);
        EXPECT_LE(env.performance(), 1.f);
    }
}
