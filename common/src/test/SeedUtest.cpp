#include "gtest/gtest.h"

#include <time.h>
#include <iostream>
#include <thread>
#include "bib/MetropolisHasting.hpp"
#include <bib/Seed.hpp>

double diffVec(const std::vector<double>& X, const std::vector<double>& Y) {
  double diff = 0;
  for (uint i = 0; i < X.size() ; i++) {
    diff += (X[i] - Y[i]) * (X[i] - Y[i]);
  }
  return sqrt(diff);
}

void random_unsafe(std::vector<double>* X) {
  srand(time(NULL));
  for (uint i = 0; i < X->size(); i++)
    X->at(i) = rand() / RAND_MAX;
}

TEST(Seed, MultiThreadUnsafeGenerator) {

  uint sampling = 5000;

  std::vector<double> X(sampling, 0);
  std::vector<double> Y(sampling, 0);

  std::thread a(random_unsafe, &X);
  std::thread b(random_unsafe, &Y);

  a.join();
  b.join();

  EXPECT_DOUBLE_EQ(diffVec(X, Y), 0.);
}


void random_safe1(std::vector<double>* X) {
  for (uint i = 0; i < X->size(); i++)
    X->at(i) = bib::Utils::rand01();
}

TEST(Seed, MultiThreadSafeGenerator) {
  uint sampling = 5000;

  std::vector<double> X(sampling, 0);
  std::vector<double> Y(sampling, 0);

  std::thread a(random_safe1, &X);
  std::thread b(random_safe1, &Y);

  a.join();
  b.join();

  EXPECT_GT(diffVec(X, Y), 20.);
}

TEST(Seed, UnifRandIntRange)
{
  for(uint n=0;n < 1000;n++){
    int r = bib::Seed::unifRandInt(3);
    
    EXPECT_GE(r, 0);
    EXPECT_LE(r, 3);
  }
  
  
}
