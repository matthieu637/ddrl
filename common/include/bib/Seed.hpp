#ifndef SEED_HPP
#define SEED_HPP

#include <iostream>
#include "Singleton.hpp"

namespace bib {

class Seed : public Singleton<Seed> {
    friend class Singleton<Seed>;

protected:
    Seed() {
        time_t t = time(0) + ::getpid();
        std::cout<<"seed: " << t << std::endl;
        srand(t);
    }

};

}


#endif
