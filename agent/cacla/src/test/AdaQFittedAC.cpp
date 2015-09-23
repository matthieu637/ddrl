#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

#include "FittedQACAg.hpp"
// #include "FittedQIterationAg.hpp"

int main(int argc, char **argv) {
  arch::Simulator<AdvancedAcrobotEnv, FittedQACAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
