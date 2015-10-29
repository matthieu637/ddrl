#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

// #include "FittedQACAg.hpp"
// #include "FittedQIterationAg.hpp"
#include "FittedNeuralACAg.hpp"
// #include "SemiFittedNeuralACAg.hpp"

int main(int argc, char **argv) {
//   arch::Simulator<AdvancedAcrobotEnv, FittedQACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, FittedQACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, FittedNeuralACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, arch::ExampleAgent> s;
//   arch::Simulator<arch::SimpleEnv1DFixedTraj, FittedNeuralACAg> s;
  arch::Simulator<AdvancedAcrobotEnv, FittedNeuralACAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
