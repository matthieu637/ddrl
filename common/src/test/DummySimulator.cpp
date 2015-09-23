
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"
#include "arch/Example.hpp"

int main(int argc, char **argv) {
  arch::Simulator<arch::ExampleEnv, arch::ExampleAgent> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
