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

TEST(AdvancedAcrobotEnv, RewardNormalized) {
  boost::property_tree::ptree properties;
  boost::property_tree::ini_parser::read_ini("data/acrobot.utest.ini", properties);

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

TEST(AdvancedAcrobotEnvSimulator, MemoryCheckDummySimulation) {
  int argc = 2;
  char config[] = "--config";
  char config_arg[] = "data/acrobot.utest.ini";
  char* argv[] = { &config[0], &config_arg[0], NULL};

  arch::Simulator<AdvancedAcrobotEnv, arch::ExampleAgent> s;
  s.init(argc, argv);

  s.run();
}


// TEST(AdvancedAcrobotEnv, RealisableByRandom) {
//   boost::property_tree::ptree properties;
//   boost::property_tree::ini_parser::read_ini("data/acrobot.utest.ini", properties);
//
//   boost::program_options::variables_map command_args;
// //   boost::program_options::options_description desc;
// //   desc.add(AdvancedAcrobotEnv::program_options());
// //   char *argv[] = {"unit-test", "--view", NULL};
// //   int argc = sizeof(argv) / sizeof(char*) - 1;
// //   boost::program_options::parsed_options parsed = boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
// //   boost::program_options::store(parsed, command_args);
// //   boost::program_options::notify(command_args);
//
//   AdvancedAcrobotEnv env;
//   env.unique_invoke(&properties, &command_args);
//   arch::ExampleAgent ag(env.number_of_actuators(), env.number_of_sensors());
//
//   bool succeded = false;
//   for (uint i = 0; i < 20000000; i++) {
//     if(i % 10000 == 0)
//       env.reset_episode();
//     env.apply(ag.run(0, env.perceptions(), false, false));
//     if(env.final_state() || env.performance() == 1.f){
//       succeded = true;
//       break;
//     }
//   }
//
//   EXPECT_EQ(succeded, true);
// }
