#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
#include "HybridCaclaAg.hpp"

int main(int argc, char **argv) {
  arch::Simulator<AdvancedAcrobotEnv, HybridCaclaAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
