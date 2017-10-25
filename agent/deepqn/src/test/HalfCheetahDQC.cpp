
#include "bib/Logger.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>

#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HalfCheetahEnv.hpp"

// #define POOL_FOR_TESTING
#include "DeepQCaclaAg.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<HalfCheetahEnv, DeepQCaclaAg<>> s;
  
  s.init(argc, argv);
  
  s.run();
  
  LOG_DEBUG("works !");
  
  google::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}
