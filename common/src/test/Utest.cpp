#include "gtest/gtest.h"
#include "nn/MLP.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
  bib::Seed::setFixedSeedUTest();
  
  uint r = RUN_ALL_TESTS();
  
  google::protobuf::ShutdownProtobufLibrary();
  gflags::ShutDownCommandLineFlags();
  return r;
}
