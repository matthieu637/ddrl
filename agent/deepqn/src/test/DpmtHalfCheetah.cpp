#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HalfCheetahEnv.hpp"
#include "DeepQNAg.hpp"
#include "nn/DevMLP.hpp"

int main(int argc, char **argv) {
  //   FLAGS_minloglevel = 2;
  //   FLAGS_logtostderr = 1;
  FLAGS_minloglevel = -4;
  FLAGS_log_dir = ".";
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<HalfCheetahEnv, DeepQNAg<>> s;
  s.init(argc, argv);
  
  s.run();
  
  LOG_DEBUG("first worked -> developping ...");
  
  //   typedef arch::DpmtAgent<CMAESAg, arch::FullyDpmtStructure, arch::FixedDpmtLearning> AgentType;
  //   typedef OCMAESAg AgentType;
  
  arch::Simulator<HalfCheetahEnv, DeepQNAg<DevMLP>> s2(1);
  s2.init(argc, argv);
  
  s.getAgent()->restoreBest();
  s2.run(s.getAgent(), s.getMaxEpisode());
  
  LOG_DEBUG("final worked");
}
