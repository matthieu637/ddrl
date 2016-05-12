#include "bib/Assert.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
#include "arch/Simulator.hpp"
#include "NeuralFittedAC.hpp"


int main(int argc, char **argv) {
  
  arch::Simulator<AdvancedAcrobotEnv, NeuralFittedAC> s;
//   arch::Simulator<arch::SimpleEnv1D, NeuralFittedAC> s;
  
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
