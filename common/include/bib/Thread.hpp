#ifndef THREAD_HPP
#define THREAD_HPP

#include <thread>
#include <map>
#include <mutex>
#include "tbb/task_scheduler_init.h"

#include "Singleton.hpp"

namespace bib {

class ThreadTBB : public Singleton<ThreadTBB> {
  friend class Singleton<ThreadTBB>;

 private:
  ThreadTBB() : current_thread_counter(0) {
    scope = new tbb::task_scheduler_init;
  }

 public:
  ~ThreadTBB() {
    delete scope;
    sleep(1); //wait thread to merge at the end of the execution
    // in order to make valgrind happy
  }
  
  uint get_number_thread(){
    return tbb::task_scheduler_init::default_num_threads();
  }
  
  size_t get_my_thread_id(){
    size_t tid = std::hash<std::thread::id>()(std::this_thread::get_id());
    auto it = ids.find(tid);
    if(it != ids.end())
      return it->second;
    
    std::lock_guard<std::mutex> guard(current_thread_mutex);
    
    uint result = current_thread_counter;
    ids.insert(std::pair<size_t, uint>(tid, current_thread_counter));
    
    current_thread_counter++;
    return result;
  }

 private:
  tbb::task_scheduler_init* scope;
  std::map<size_t, uint> ids;
  std::mutex current_thread_mutex;
  uint current_thread_counter;
};

}

#endif
