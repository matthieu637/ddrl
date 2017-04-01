
#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HumanoidEnv.hpp"

int main(int argc, char **argv) {
//   arch::Simulator<HumanoidEnv, arch::ExampleAgent> s;
  arch::Simulator<HumanoidEnv, arch::ZeroAgent> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
