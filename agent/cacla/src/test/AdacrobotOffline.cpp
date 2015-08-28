#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
// #include "OfflineCaclaAg.hpp"
// #include "OfflineCaclaAg2.hpp"
#include "OfflineCaclaAg3.hpp"
// #include "OfflineCaclaAg3Aon.hpp"
// #include "OfflineCaclaAg4.hpp"
// #include "OfflineCaclaAg4ac.hpp"

int main(int argc, char **argv) {
  arch::Simulator<AdvancedAcrobotEnv, OfflineCaclaAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
