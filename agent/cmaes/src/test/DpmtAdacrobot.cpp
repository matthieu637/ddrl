#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"
#include "CMAESAg.hpp"
#include "OCMAESAg.hpp"
#include "arch/DpmtAgent.hpp"

int main(int argc, char **argv) {
  arch::Simulator<AdvancedAcrobotEnv, CMAESAg> s;
  s.init(argc, argv);

  s.run();
  
  LOG_DEBUG("first worked -> developping ...");
  
//   typedef arch::DpmtAgent<CMAESAg, arch::FullyDpmtStructure, arch::FixedDpmtLearning> AgentType;
  typedef OCMAESAg AgentType;
  
  arch::Simulator<AdvancedAcrobotEnv, AgentType> s2(1);
  s2.init(argc, argv);

  s.getAgent()->restoreBest();
  s2.run(s.getAgent(), s.getMaxEpisode());
  
  LOG_DEBUG("final worked");
}
