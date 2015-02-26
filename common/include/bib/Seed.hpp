#ifndef SEED_HPP
#define SEED_HPP

#include <iostream>
#include <thread>
#include <map>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "Singleton.hpp"

#define RAND() : Seed::getInstace()->rand();

namespace bib {

class Seed : public Singleton<Seed> {
  friend class Singleton<Seed>;

 protected:
  Seed() {}

 public:
  unsigned int* getSeed() {
    std::thread::id my_id = std::this_thread::get_id();
//         tbb::tbb_thread::id my_id_tbb = tbb::this_tbb_thread::get_id();

    if(seeds.find(my_id) != seeds.end()) {
      return &(seeds[my_id]);
    } else return createSeed(my_id);
  }

  int rand() {
    return rand_r(getSeed());
  }

 private:
  unsigned int* createSeed(const std::thread::id& my_id) {
    boost::unique_lock< boost::shared_mutex > w_lock(_write_mutex);
    size_t id_hash = std::hash<std::thread::id>()(my_id);

    std::pair<std::thread::id, unsigned int> element(my_id, time(0) + id_hash);
    std::map< std::thread::id, unsigned int>::iterator it = seeds.insert(element).first;
    LOG_DEBUG("new seed " << element.second << " " << my_id);
    return &(it->second);
  }
//     TODO:auto clear map?

 private:
  std::map< std::thread::id, unsigned int> seeds;
  boost::shared_mutex _write_mutex;

};

}

#endif
