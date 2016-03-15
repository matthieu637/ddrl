#include "bib/Assert.hpp"
#include "arch/ACSimulator.hpp"
#include "arch/Example.hpp"
#include "CartpoleEnv.hpp"
#include "OffPolSetACFitted.hpp"

int main(int argc, char **argv) {
  arch::ACSimulator<CartpoleEnv, OffPolSetACFitted> s;
  
  s.init(argc, argv);
  s.enable_analyse_distance_bestVF();

  s.run();

  LOG_DEBUG("works !");
}
