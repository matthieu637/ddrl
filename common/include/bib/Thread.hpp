#ifndef THREAD_HPP
#define THREAD_HPP

#include "tbb/task_scheduler_init.h"

#include "Singleton.hpp"

namespace bib {

class ThreadTBB : public Singleton<ThreadTBB> {
  friend class Singleton<ThreadTBB>;

 private:
  ThreadTBB() {
    scope = new tbb::task_scheduler_init;
  }

 public:
  ~ThreadTBB() {
    delete scope;
    sleep(1); //wait thread to merge at the end of the execution
    // in order to make valgrind happy
  }

 private:
  tbb::task_scheduler_init* scope;
};

}

#endif
