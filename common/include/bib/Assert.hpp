#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <iostream>

// #ifdef _ASSERT_H
// #warning do not use assert, please use our own assert method in order to speed up simulation
// #endif

// avoid the inclusion of assert.h
// #define _ASSERT_H

#ifndef NDEBUG
#define ASSERT(condition, stream)                                             \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cout << "#ASSERT FAILED :" << __FILE__ << "." << __LINE__ << " : " \
                << stream << std::endl;                                       \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#define _ASSERT_EQ(val1, val2)                                                 \
  do {                                                                        \
    if (val1 != val2) {                                                       \
      std::cout << "#ASSERT FAILED :" << __FILE__ << "." << __LINE__ << " : " \
                << val1 << " " << val2 << std::endl;                          \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#define _ASSERT_EQS(val1, val2, stream)                                        \
  do {                                                                        \
    if (val1 != val2) {                                                       \
      std::cout << "#ASSERT FAILED :" << __FILE__ << "." << __LINE__ << " : " \
                << val1 << " " << val2 << " " << stream << std::endl;         \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#else
#define ASSERT(condition, stream)
#define _ASSERT_EQ(val1, val2)
#define _ASSERT_EQS(val1, val2, stream)
#endif

#endif
