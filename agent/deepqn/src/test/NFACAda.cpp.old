#include "bib/Logger.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>

#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

#include "NeuralFittedACAg.hpp"

int main(int argc, char **argv) {
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  arch::Simulator<AdvancedAcrobotEnv, NeuralFittedACAg> s;

  s.init(argc, argv);

  s.run();

  google::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  google::protobuf::ShutdownProtobufLibrary();
  
  return 0;
}
