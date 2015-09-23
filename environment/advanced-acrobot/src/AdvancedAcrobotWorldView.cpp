#include "AdvancedAcrobotWorldView.hpp"

#include <string>
#include <vector>
#include "boost/filesystem.hpp"

#include "Draw.hpp"
#include "bib/Logger.hpp"

static AdvancedAcrobotWorldView* inst = nullptr;

void parseCommand(int cmd) {

  static float xyz[3] = {0., -3., 1};
  static float hpr[3] = {90, 0, 0};

  switch (cmd) {
  case 'f':
    inst->speed = inst->speed * 2.;
    if(inst->speed > 16)
      inst->speed=16;
    LOG_DEBUG("speed : x " <<inst->speed);
    break;
  case 'd':
    inst->speed = inst->speed / 2.;
    if(inst->speed < 0.5)
      inst->speed = 0.5;
    LOG_DEBUG("speed : x " <<inst->speed);
    break;
  case 'a':
    inst->ignoreMotor = !(inst->ignoreMotor);
    if (inst->ignoreMotor)
      LOG_DEBUG("motor ignored");
    else
      LOG_DEBUG("motor powered");
    break;
  case 'i':
    dsSetViewpoint(xyz, hpr);
    break;
  case 's':
    bib::Logger::PRINT_ELEMENTS_FT(inst->state(), "STATE : ", 6, 2);
    LOG_DEBUG("PERFORMANCE " << inst->perf());
    break;
  case 'v':
    float vxyz[3];
    float vhpr[3];
    dsGetViewpoint(vxyz, vhpr);
    LOG_DEBUG("view point : " << vxyz[0] << " " << vxyz[1] << " " << vxyz[2]
              << " " << vhpr[0] << " " << vhpr[1] << " "
              << vhpr[2]);
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

  static float xyz[3] = {0., -3., 1};
  static float hpr[3] = {90, 0, 0};
  dsSetViewpoint(xyz, hpr);

  while (!inst->requestEnd) {
    HACKdraw(&inst->fn);
    //wait time between frame draw
    usleep(1 * 1000);
  }
}

AdvancedAcrobotWorldView::AdvancedAcrobotWorldView(
  const std::string& path, const std::vector<bone_joint>& types,
  const std::vector<bool>& actuators, bool _add_time_in_state, bool normalization)
  : AdvancedAcrobotWorld(types, actuators, _add_time_in_state, normalization),
    requestEnd(false), 
    speed(1),
    ignoreMotor(false) {
  std::string goodpath = path;

  int n;
  for (n = 0; n < 5; n++)
    if (!boost::filesystem::exists(goodpath)) {
      LOG_DEBUG(goodpath << " doesnt exists");
      goodpath = std::string("../") + goodpath;
    } else {
      break;
    }

  if (n >= 5) {
    LOG_ERROR("cannot found " << path);
    exit(1);
  }
  inst = this;

  for (ODEObject * b : bones)
    geoms.push_back(b->getGeom());

//     ODEObject* debug1= ODEFactory::getInstance()->createBox(
//                         odeworld, 0.5, 0, 0.5/2.f, BONE_LARGER, BONE_LARGER, 0.5,
//                         BONE_DENSITY, BONE_MASS, false);
//     ODEObject* debug2= ODEFactory::getInstance()->createBox(
//                         odeworld, 1., 0, 1.1/2.f, BONE_LARGER, BONE_LARGER, 1.1,
//                         BONE_DENSITY, BONE_MASS, false);
//
//     dGeomSetPosition(debug1->getGeom(), debug1->getX(), debug1->getY(), debug1->getZ());
//     dGeomSetPosition(debug2->getGeom(), debug2->getX(), debug2->getY(), debug2->getZ());
//
//     geoms.push_back(debug1->getGeom());
//     geoms.push_back(debug2->getGeom());

  eventThread = new tbb::tbb_thread(threadloop, goodpath);
}

AdvancedAcrobotWorldView::~AdvancedAcrobotWorldView() {
  //     for(auto it=delete_me_later.begin(); it != delete_me_later.end(); ++it)
  //     {
  //         dGeomDestroy((*it)->getGeom());
  //         delete *it;
  //     }

  requestEnd = true;
  eventThread->join();
  delete eventThread;
  HACKclose();
}

void AdvancedAcrobotWorldView::step(const std::vector<float>& motors, uint current_step, uint max_step_per_instance) {
  std::vector<float> modified_motors(motors.size(), 0);
  if (!inst->ignoreMotor) {
    for (unsigned int i = 0; i < motors.size(); i++)
      modified_motors[i] = motors[i];
  }

  AdvancedAcrobotWorld::step(modified_motors, current_step, max_step_per_instance);

  // approximative human vision smooth
  // usleep(25 * 1000);

  usleep((25 / speed)  * 1000);  // needed to don't be faster than the view
}
