#include "bib/Seed.hpp"


#include <time.h>
#include <thread>

namespace bib {

thread_local std::mt19937 bib::Seed::engine(std::clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
bib::Seed bib::Seed::seed_instance;
}
