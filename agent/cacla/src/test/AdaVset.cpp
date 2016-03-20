#include "bib/Assert.hpp"
#include "arch/ACSimulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
#include "OffVSetACFitted.hpp"

int main(int argc, char **argv) {
  arch::ACSimulator<AdvancedAcrobotEnv, OffVSetACFitted> s;
  
  s.init(argc, argv);
  s.enable_analyse_distance_bestPol();

  s.run();

  LOG_DEBUG("works !");
}
