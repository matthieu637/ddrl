#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

#include "ContinuousAcAg.hpp"
#include "ContinuousAcTAg.hpp"

int main(int argc, char **argv) {
//   arch::Simulator<AdvancedAcrobotEnv, ContinuousAcAg> s;
//   arch::Simulator<AdvancedAcrobotEnv, ContinuousAcTAg, arch::MotorEpisodeStat> s;
  arch::Simulator<AdvancedAcrobotEnv, ContinuousAcTAg> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
