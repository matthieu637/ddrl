#include "gtest/gtest.h"
#include "arch/Simulator.hpp"
#include "arch/Example.hpp"

TEST(Simulator, MemoryCheckDummy)
{
    arch::Simulator<arch::ExampleEnv, arch::ExampleAgent> s;
    s.init();

    s.run();
}
