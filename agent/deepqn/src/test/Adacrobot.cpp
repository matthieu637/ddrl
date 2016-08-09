#include "bib/Logger.hpp"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>

#include "bib/Assert.hpp"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "AdvancedAcrobotEnv.hpp"

#include "DeepQNAg.hpp"

int main(int argc, char **argv) {
//   std::string usage(argv[0]);
//   usage.append(" -[evaluate|save [path]]");
//   gflags::SetUsageMessage(usage);
//   gflags::SetVersionString("0.1");
//   gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  arch::Simulator<AdvancedAcrobotEnv, DeepQNAg> s;

  s.init(argc, argv);

  s.run();

  LOG_DEBUG("works !");
}
