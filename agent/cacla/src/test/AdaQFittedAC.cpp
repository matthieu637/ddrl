#include "bib/Assert.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
//#include "arch/Simulator.hpp"
#include "arch/ACSimulator.hpp"

// #include "FittedQACAg.hpp"
// #include "FittedQIterationAg.hpp"
// #include "FittedNeuralACAg.hpp"
// #include "SemiFittedNeuralACAg.hpp"

//#include "OfflineCaclaAgLW.hpp"
//#include "OfflineCaclaAg.hpp"
#include "OffPolSetACFitted.hpp"


int main(int argc, char **argv) {
//   arch::Simulator<AdvancedAcrobotEnv, FittedQACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, FittedQACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, FittedNeuralACAg> s;
//   arch::Simulator<arch::SimpleEnv1D, arch::ExampleAgent> s;
//   arch::Simulator<arch::SimpleEnv1DFixedTraj, FittedNeuralACAg> s;
//   arch::Simulator<AdvancedAcrobotEnv, FittedNeuralACAg, arch::MotorEpisodeStat> s;
//  arch::Simulator<AdvancedAcrobotEnv, FittedNeuralACAg> s;
  arch::ACSimulator<AdvancedAcrobotEnv, OffPolSetACFitted> s;
  
  s.init(argc, argv);
  s.enable_analyse_distance_bestVF();

  s.run();

  LOG_DEBUG("works !");
}
