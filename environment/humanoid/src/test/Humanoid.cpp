#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HumanoidEnv.hpp"
#include <google/protobuf/stubs/common.h>
// #include <gflags/gflags.h>

int main(int argc, char **argv) {
  arch::Simulator<HumanoidEnv, arch::ExampleAgent> s;
//   arch::Simulator<HumanoidEnv, arch::ZeroAgent> s;
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
  
  google::protobuf::ShutdownProtobufLibrary();
//   gflags::ShutDownCommandLineFlags();
  return 0;
}
