#include <ODEFactory.hpp>
#include "AdvancedAcrobotWorld.hpp"
#include "bib/Utils.hpp"


int main(int, char **) {

    ODEFactory::getInstance();

    AdvancedAcrobotWorld* simu1 = new AdvancedAcrobotWorld;

    AdvancedAcrobotWorld* simu2 = new AdvancedAcrobotWorld;

//     simu1->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//     simu2->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//     simu2->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//     simu2->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//
//     simu1->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//     simu1->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());
//
//     simu2->step(bib::Utils::rand01(), bib::Utils::rand01(), bib::Utils::rand01());

    delete simu2;
    delete simu1;

    ODEFactory::endInstance();

    return 0;
}
