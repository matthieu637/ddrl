#ifndef ADVANCEDACROBOTWORLDVIEW_HPP
#define ADVANCEDACROBOTWORLDVIEW_HPP

#include <AdvancedAcrobotWorld.hpp>
#include <string>
#include <tbb/tbb.h>
#include "drawstuff.h"

class AdvancedAcrobotWorldView : public AdvancedAcrobotWorld
{
public:
    AdvancedAcrobotWorldView(const std::string&);
    ~AdvancedAcrobotWorldView();
    void step(const std::vector<float>& motors);

public:
    std::list<dGeomID> geoms;
//     std::list<ODEObject*> delete_me_later;
    
    tbb::tbb_thread* eventThread;
    dsFunctions fn;
    bool requestEnd;
    
    //specific keyboard behavior
    bool speedUp;
    bool ignoreMotor;
};


#endif // ADVANCEDACROBOTWORLDVIEW_HPP
