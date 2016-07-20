#include "bib/Logger.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>

#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "CartpoleEnv.hpp"

#include "DeepQNAg.hpp"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<CartpoleEnv, DeepQNAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
