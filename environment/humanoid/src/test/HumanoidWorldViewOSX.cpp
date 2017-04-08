#include "HumanoidWorldView.hpp"

#include <string>
#include <vector>
#include <thread>
#include "boost/filesystem.hpp"

#include "Draw.hpp"
#include "bib/Logger.hpp"

#include "arch/Simulator.hpp"
#include "arch/Example.hpp"
#include "HumanoidEnv.hpp"

static HumanoidWorldView* inst = nullptr;
static std::string goodpath;

void parseCommandHumanoid(int cmd) {
  static float xyz[3] = {-0.03, -0.97, 0.2};
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
    if(inst->speed < 0.0625)
      inst->speed = 0.0625;
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
    bib::Logger::PRINT_ELEMENTS_FT(inst->state(), "STATE : ", 8, 2);
    LOG_DEBUG("REWARD : " << inst->performance());
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
    inst->resetPositionsView();
    LOG_DEBUG("resetPositions should not be used");
    break;
  case 'x':
    inst->modified_motor = -1.f;
    LOG_DEBUG("motors applied");
    break;
  case 'w':
    inst->modified_motor = 1.f;
    LOG_DEBUG("motors applied");
    break;
  }
}

void threadloopHumanoid(const std::string& goodpath) {
  ASSERT(inst != nullptr, "not instantiated " << goodpath);
  inst->fn.version = DS_VERSION;
  inst->fn.start = 0;
  inst->fn.step = &Draw::drawLoop;
  inst->fn.command = &parseCommandHumanoid;
  inst->fn.stop = 0;
  inst->fn.path_to_textures = goodpath.c_str();

  Draw::geoms = &inst->bones;

  HACKinitDs(1280, 720, &inst->fn);

  float xyz[3] = {-3.445, 0, 1.};
  float hpr[3] = {3.5, -9, -1.3};
  dsSetViewpoint(xyz, hpr);

  while (!inst->requestEnd) {
    Mutex::scoped_lock lock(inst->mutex_reset);
    HACKdraw(&inst->fn);
    lock.release();
    //wait time between frame draw
//     usleep(1 * 1000);//each milisecond -> 1000fps
    usleep(10 * 1000);//each milisecond -> 100fps
    
//     double x = dBodyGetPosition(inst->bones[1]->getID())[0];
//     dsGetViewpoint(xyz, hpr);
//     xyz[0] = x - 1.43;
//     dsSetViewpoint(xyz, hpr);
  }
}

HumanoidWorldView::HumanoidWorldView(const std::string& path, const humanoid_physics phy)
  : HumanoidWorld(phy),
    requestEnd(false),
    speed(1.),
    ignoreMotor(false) {
  goodpath = path;

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

//   for (ODEObject * b : bones)
//     geoms.push_back(b->getGeom());

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
}

HumanoidWorldView::~HumanoidWorldView() {
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

void HumanoidWorldView::step(const std::vector<double>& motors) {
  LOG_DEBUG("step2");
  std::vector<double> modified_motors(motors.size(), 0);
  if (!inst->ignoreMotor) {
    if(inst->modified_motor <= -2.f)
      for (unsigned int i = 0; i < motors.size(); i++)
        modified_motors[i] = motors[i];
    else 
      for (unsigned int i = 0; i < motors.size(); i++)
        modified_motors[i] = inst->modified_motor;
  }

  Mutex::scoped_lock lock(mutex_reset);
  HumanoidWorld::step(modified_motors);
  lock.release();

  inst->modified_motor = -2.f;
  // approximative human vision smooth
//   usleep(25 * 1000);
  

  usleep(3*WORLD_STEP / speed * 1000 * 1000);  // needed to don't be faster than the view
  LOG_DEBUG("step");
}

void HumanoidWorldView::resetPositions(std::vector<double> & result_stoch, const std::vector<double>& given_stoch) {
  Mutex::scoped_lock lock(mutex_reset);
  HumanoidWorld::resetPositions(result_stoch, given_stoch);
  lock.release();
}

void HumanoidWorldView::resetPositionsView() {
  std::vector<double> qq;
  HumanoidWorld::resetPositions(qq, qq);
}

//static arch::Simulator<HumanoidEnv, arch::ExampleAgent>* s;
static arch::Simulator<HumanoidEnv, arch::ZeroAgent>* s;

void threadOSXRun(){
  LOG_DEBUG("threadOSX start");
  sleep(2);
  s->run_loop(0, 100);
  LOG_DEBUG("threadOSX finished");
}

int main(int argc, char **argv) {
  //s = new arch::Simulator<HumanoidEnv, arch::ExampleAgent>();
  s = new arch::Simulator<HumanoidEnv, arch::ZeroAgent>();
  s->init(argc, argv);
  s->before_run((arch::ExampleAgent*) nullptr, 0);
    
  std::thread t2(threadOSXRun);
  threadloopHumanoid(goodpath);
  t2.join();

  LOG_DEBUG("works !");
  delete s;
}


