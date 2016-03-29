#include "bib/Assert.hpp"
#include "arch/ACSimulator.hpp"
#include "arch/Example.hpp"
#include "CartpoleEnv.hpp"
#include "OffVSetACFitted.hpp"

int main(int argc, char **argv) {
  arch::ACSimulator<CartpoleEnv, OffVSetACFitted> s;
  
  s.init(argc, argv);
  s.enable_analyse_distance_bestPol();

  s.run();

  LOG_DEBUG("works !");
}
