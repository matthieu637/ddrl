#include "bib/Seed.hpp"


#include <time.h>
#include <thread>

namespace bib {

// thread_local std::mt19937 bib::Seed::engine(std::clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
thread_local std::shared_ptr<bib::Seed> Seed::seed_instance(new bib::Seed);
}
