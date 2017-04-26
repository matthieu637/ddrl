#ifndef UTEST_HPP
#define UTEST_HPP

#include <gflags/gflags.h>
#include "gtest/gtest.h"

DECLARE_bool(valgrind);

#define VALGRIND_REDUCE(m,n)\
(FLAGS_valgrind ? n : m)

#define VALGRIND_SKIP(m)    \
do {                        \
  if (!(FLAGS_valgrind)) {  \
    m;                       \
  }                         \
} while (0)

#endif
