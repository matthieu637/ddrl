#include "gtest/gtest.h"
#include "nn/MLP.hpp"
#include "bib/UTest.hpp"

DEFINE_bool(valgrind, false, "Speed-up test for memory test only");

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;

  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
  bib::Seed::setFixedSeedUTest();
  
  if(FLAGS_valgrind){
    LOG_INFO("valgrind captured");
  }
  
  uint r = RUN_ALL_TESTS();
  
  google::protobuf::ShutdownProtobufLibrary();
  gflags::ShutDownCommandLineFlags();
  return r;
}
