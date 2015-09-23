
#include "AdvancedAcrobotWorldView.hpp"
#include <boost/filesystem.hpp>
#include "Draw.hpp"
#include "bib/Logger.hpp"

static AdvancedAcrobotWorldView* inst = nullptr;

void parseCommand(int cmd) {
    static float xyz[3] = {0.,-3.,1};
    static float hpr[3] = {90, 0,0};

    switch (cmd) {
    case 'f':
        inst->speedUp = true;
        break;
    case 'd':
        inst->speedUp = false;
        break;
    case 'a':
        inst->ignoreMotor=!(inst->ignoreMotor);
        if(inst->ignoreMotor)
            LOG_DEBUG("motor ignored");
        else
            LOG_DEBUG("motor powered");
        break;
    case 'i':
        dsSetViewpoint (xyz,hpr);
        break;
    case 's':
g        bib::Logger::PRINT_ELEMENTS_FT(inst->state(), "STATE : ",6,2);
        LOG_DEBUG("PERFORMANCE " << inst->perf());
        break;
    case 'v':
        float vxyz[3];
        float vhpr[3];
        dsGetViewpoint(vxyz, vhpr);
        LOG_DEBUG("view point : " << vxyz[0] << " "<<  vxyz[1] << " "<< vxyz[2] << " " << vhpr[0] << " " << vhpr[1] << " "<< vhpr[2] );
        break;
    case 'r':
        inst->resetPositions();
        LOG_DEBUG("resetPositions should not be used");
        break;
    }
}

void threadloop(const std::string& goodpath) {
    ASSERT(inst != nullptr, "not instantiated " << goodpath);
    inst->fn.version = DS_VERSION;
    inst->fn.start = 0;
    inst->fn.step = &Draw::drawLoop;
    inst->fn.command = &parseCommand;
    inst->fn.stop = 0;
    inst->fn.path_to_textures = goodpath.c_str();

    Draw::geoms = &inst->geoms;

    HACKinitDs(1280, 720, &inst->fn);

    static float xyz[3] = {0.,-3.,1};
    static float hpr[3] = {90, 0,0};
    dsSetViewpoint (xyz,hpr);

    while(!inst->requestEnd)
    {
        HACKdraw(&inst->fn);
        usleep(10 *1000);
    }
}

AdvancedAcrobotWorldView::AdvancedAcrobotWorldView(const std::string& path) : AdvancedAcrobotWorld(), requestEnd(false), speedUp(false), ignoreMotor(false)
{
    std::string goodpath = path;

    int n;
    for(n=0; n<5; n++)
        if(!boost::filesystem::exists(goodpath)) {
            LOG_DEBUG(goodpath << " doesnt exists");
            goodpath = std::string("../") + goodpath;
        }
        else break;

    if(n >= 5) {
        LOG_ERROR("cannot found " << path);
        exit(1);
    }
    inst = this;

    for(ODEObject* b : bones)
        geoms.push_back(b->getGeom());


    eventThread = new tbb::tbb_thread(threadloop,goodpath);
}

AdvancedAcrobotWorldView::~AdvancedAcrobotWorldView()
{
//     for(auto it=delete_me_later.begin(); it != delete_me_later.end(); ++it) {
//         dGeomDestroy((*it)->getGeom());
//         delete *it;
//     }

    requestEnd=true;
    eventThread->join();
    delete eventThread;
    HACKclose();
}


void AdvancedAcrobotWorldView::step(const std::vector<float>& motors)
{
    std::vector<float> modified_motors(motors.size(), 0);
    if(!inst->ignoreMotor) {
        for(unsigned int i=0; i<motors.size(); i++)
            modified_motors[i] = motors[i];
    }

    AdvancedAcrobotWorld::step(modified_motors);

    if(!speedUp)
        usleep(250*1000);

    usleep(20*1000);//needed to don't be faster than the view
}

