#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HumanoidEnv.hpp"
#include "DeepQNAg.hpp"
#include "nn/DODevMLP.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  //   FLAGS_logtostderr = 1;
  //FLAGS_minloglevel = -4;
  //FLAGS_log_dir = ".";
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<HumanoidEnv, DeepQNAg<DODevMLP>> s;
  s.init(argc, argv);
  
  s.run();
  
  LOG_DEBUG("final worked");
}
