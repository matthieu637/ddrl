#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <iostream>

#ifdef _ASSERT_H
#warning do not use assert, please use our own assert method in order to speed up simulation
#endif

// avoid the inclusion of assert.h
#define _ASSERT_H

#ifndef NDEBUG
#define ASSERT(condition, stream)                                             \
  do {                                                                        \
    if (!(condition)) {                                                       \
      std::cout << "#ASSERT FAILED :" << __FILE__ << "." << __LINE__ << " : " \
                << stream << std::endl;                                       \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#else
#define ASSERT(condition, stream)
#endif

#endif
