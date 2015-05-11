#include "gtest/gtest.h"
#include <bib/Thread.hpp>

int main(int argc, char **argv) {
  bib::ThreadTBB::getInstance();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
