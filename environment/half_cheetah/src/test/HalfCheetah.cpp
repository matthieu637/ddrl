
#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HalfCheetahEnv.hpp"

int main(int argc, char **argv) {
  arch::Simulator<HalfCheetahEnv, arch::ExampleAgent> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
