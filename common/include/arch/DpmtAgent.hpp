#ifndef DPMTAGENT_HPP
#define DPMTAGENT_HPP

#include <vector>
#include <string>

#include "arch/Dummy.hpp"
#include "arch/CommonAE.hpp"
#include "AACAgent.hpp"

namespace arch {
  
class DpmtStructure {
  
};

class FullyDpmtStructure : public DpmtStructure {
  
};

class DpmtLearning {
  
};

class FixedDpmtLearning : public DpmtLearning{
  
};

template <typename OAgent, typename DpmtStructureImpl, typename DpmtLearningImpl, typename ProgOptions = AgentProgOptions>
class DpmtAgent : public OAgent {
//   static_assert(std::is_base_of<AACAgent<ProgOptions>, OAgent>::value, "Agent should be a base of AACAgent.");
//   static_assert(std::is_base_of<DpmtStructure, DpmtStructureImpl>::value, "DpmtStructureImpl should be a base of DpmtStructure.");
//   static_assert(std::is_base_of<DpmtLearning, DpmtStructureImpl>::value, "DpmtStructureImpl should be a base of DpmtLearning.");
  
 public:
  DpmtAgent(uint _nb_motors, uint _nb_sensors): OAgent(_nb_motors, _nb_sensors){
    
  }
  
  void provide_early_development(AAgent<>* _old_ag) override {
    old_ag= static_cast<OAgent*>(_old_ag);
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& perceptions,
                                        bool learning, bool goal_reached, bool finished) override{
                                          
  };
  
  OAgent* old_ag = nullptr;
};
}  // namespace arch

#endif  // DPMTAGENT_HPP
