#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "CartpoleEnv.hpp"
#include "BaseCaclaAg.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<CartpoleEnv, BaseCaclaAg> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
