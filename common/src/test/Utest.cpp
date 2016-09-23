#include "gtest/gtest.h"
#include "nn/MLP.hpp"

int main(int argc, char **argv) {
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
