#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HalfCheetahEnv.hpp"
#include "nn/DODevMLP.hpp"
#include "CMAESAg.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<HalfCheetahEnv, CMAESAg<DODevMLP>> s;
  
  s.init(argc, argv);
  
  s.run();
  
  LOG_DEBUG("works !");
}
