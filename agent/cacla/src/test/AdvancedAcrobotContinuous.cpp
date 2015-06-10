#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
// #include "BaseCaclaAg.hpp"
#include "OfflineCaclaAg.hpp"

int main(int argc, char **argv) {
  arch::Simulator<AdvancedAcrobotEnv, OfflineCaclaAg> s;
//   arch::Simulator<AdvancedAcrobotEnv, BaseCaclaAg> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
