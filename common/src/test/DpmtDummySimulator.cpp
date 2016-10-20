
#include "arch/Simulator.hpp"
#include "bib/Logger.hpp"
#include "arch/Example.hpp"


class GreaterExampleEnv : public arch::AEnvironment<> {
 public:
  GreaterExampleEnv() : internal_state(8) {
    for (unsigned int i = 0; i < internal_state.size(); i++)
      internal_state[i] = bib::Utils::randin(-1, 1);
  }
  const std::vector<double>& perceptions() const {
    return internal_state;
  }
  unsigned int number_of_actuators() const {
    return 5;
  }
  unsigned int number_of_sensors() const {
    return internal_state.size();
  }
  double performance() const {
    return 0;
  }
  void _apply(const std::vector<double>&) {}

  std::vector<double> internal_state;
};

class DpmtExampleAgent : public arch::AAgent<> {
 public:
  DpmtExampleAgent(unsigned int nb_motors, unsigned int) : actuator(nb_motors) {}
  const std::vector<double>& run(double, const std::vector<double>&, bool, bool) override {
    time_for_ac--;
    if (time_for_ac == 0) {
      time_for_ac = decision_each;
      for (unsigned int i = 0; i < actuator.size(); i++)
        actuator[i] = bib::Utils::randin(-1, 1);
    }
    return actuator;
  }
  
  void start_episode(const std::vector<double>&, bool) override {
    time_for_ac = 1;
  }

  virtual ~DpmtExampleAgent() {
  }
  
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    decision_each                       = pt->get<int>("agent.decision_each");
  }
  
  void provide_early_development(AAgent* _old_ag) override {
    old_ag= static_cast<arch::ExampleAgent*>(_old_ag);
  }

  int decision_each, time_for_ac;
  std::vector<double> actuator;
  arch::ExampleAgent* old_ag = nullptr;
};

int main(int argc, char **argv) {
  arch::Simulator<arch::ExampleEnv, arch::ExampleAgent> s(0);
  s.init(argc, argv);

  s.run();

  LOG_DEBUG("first worked -> developping ...");
  
  arch::Simulator<GreaterExampleEnv, DpmtExampleAgent> s2(1);
  s2.init(argc, argv);

  s2.run(s.getAgent(), s.getMaxEpisode());
  
  LOG_DEBUG("final worked");
}
