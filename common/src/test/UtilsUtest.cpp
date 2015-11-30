#include "gtest/gtest.h"
#include "bib/Utils.hpp"
#include <bib/Seed.hpp>

TEST(Utils, random) {
  for (uint i = 0; i < 100; i++) {
    double f = bib::Utils::rand01();

    EXPECT_GE(f, 0);
    EXPECT_LE(f, 1);
  }
}

TEST(Utils, transform) {
  for (uint i = 0; i < 100; i++) {
    double f = bib::Utils::rand01();
    double _min = bib::Utils::rand01();
    double _max = bib::Utils::rand01();

    if (_min > _max)
      _min = _max;

    f = bib::Utils::transform(f, 0.f, 1.f, _min, _max);
    EXPECT_GE(f, _min);
    EXPECT_LE(f, _max);
  }
}
