#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "PowerAg.hpp"
#include "Kernel.hpp"
#include "AdvancedAcrobotEnv.hpp"
#include <iostream>

int main(int argc, char **argv) {

  arch::Simulator<AdvancedAcrobotEnv, PowerAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
