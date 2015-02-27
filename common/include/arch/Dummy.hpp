#ifndef DUMMY_H
#define DUMMY_H

#include <boost/program_options.hpp>

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
} // namespace arch

#endif  // DUMMY_H
