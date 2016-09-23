#ifndef DUMMY_H
#define DUMMY_H

#include <string>
#include "boost/program_options.hpp"

#include "bib/Logger.hpp"
#include "bib/Prober.hpp"

namespace arch {

class DummyProgOptions {
 public:
  static boost::program_options::options_description program_options() {
    boost::program_options::options_description desc;
    return desc;
  }
};

class AgentProgOptions {
 public:
  static boost::program_options::options_description program_options() {
    boost::program_options::options_description desc("Allowed Agent options");
    desc.add_options()("load", boost::program_options::value<std::string>(),
                       "set the agent to load");
    return desc;
  }
};

class EnvProgOptions {
 public:
  static boost::program_options::options_description program_options() {
    boost::program_options::options_description desc(
      "Allowed environment options");
    desc.add_options()("view", "display the environment [default : false]");
    return desc;
  }
};


class DummyEpisodeStat {
 public:
  virtual void dump(uint episode, const std::vector<double>& perceptions, const std::vector<double>& motors, double reward) {
    (void) episode;
    (void) perceptions;
    (void) motors;
    (void) reward;
  }
};

class MotorEpisodeStat : public DummyEpisodeStat {
 public:

  MotorEpisodeStat() : step(0) { }

  virtual void dump(uint episode, const std::vector<double>& perceptions, const std::vector<double>& motors, double reward) {
    (void) episode;
    (void) perceptions;
    (void) reward;

    std::string sep = std::to_string(episode);;
    LOG_FILE_NNL("motors.data." + sep, step << " ");
    for(double m : motors)
      LOG_FILE_NNL("motors.data." + sep, m << " ");
    LOG_FILE("motors.data." + sep, "");

    step++;
  }

 private :
  uint step;
};

class PerceptionEpisodeStat : public DummyEpisodeStat {
 public:

  PerceptionEpisodeStat() : step(0) { }

  virtual void dump(uint episode, const std::vector<double>& perceptions, const std::vector<double>& motors, double reward) {
    (void) episode;
    (void) motors;
    (void) reward;

    std::string sep = std::to_string(episode);;
    LOG_FILE_NNL("perceptions.data." + sep, step << " ");
    for(double m : perceptions)
      LOG_FILE_NNL("perceptions.data." + sep, m << " ");
    LOG_FILE("perceptions.data." + sep, "");

    step++;
  }

 private :
  uint step;
};

class PerceptionProbStat : public DummyEpisodeStat {
 public:

  PerceptionProbStat() : first(true), probes(0), step(0)  { }

  virtual void dump(uint episode, const std::vector<double>& perceptions, const std::vector<double>& motors, double reward) {
    (void) episode;
    (void) motors;
    (void) reward;
    
    if(first){
      probes.resize(perceptions.size()); 
      first = false;
    }
    
    auto si = perceptions.begin();
    for(auto p = probes.begin() ; p != probes.end() ;++p){
      p->probe(*si);
      si++;
    }
    
    step++;
    if(step % 10000 == 0){
      for(auto p : probes){
        LOG_FILE_NNL("perceptions_probe.data", p.min_probe << " " << p.max_probe << " ");
      }
      LOG_FILE("perceptions_probe.data", "");
    }
  }

 private :
  bool first;
  std::vector<bib::Prober> probes;
  uint step;
};

class AgentGPUProgOptions {
 public:
  static boost::program_options::options_description program_options() {
    boost::program_options::options_description desc("Allowed Agent options");
    desc.add_options()("load", boost::program_options::value<std::string>(), "set the agent to load");
    desc.add_options()("cpu", "use cpu [default]");
    desc.add_options()("gpu", "use gpu");
    return desc;
  }
};

}  // namespace arch

#endif  // DUMMY_H
