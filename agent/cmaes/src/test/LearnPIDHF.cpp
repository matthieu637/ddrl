#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HalfCheetahEnv.hpp"
#include "PIDControllerLearn.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<HalfCheetahEnv, PIDControllerLearn> s;
  
  s.init(argc, argv);
  
  s.run();
  
  PIDControllerLearn* ag = static_cast<PIDControllerLearn*>(s.getAgent());
  ag->print_final_best();
  
  LOG_DEBUG("works !");
}
