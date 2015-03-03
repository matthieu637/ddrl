#ifndef AAGENT_H
#define AAGENT_H

#include <vector>
#include <string>

#include "arch/Dummy.hpp"
#include "arch/CommonAE.hpp"

namespace arch {

template <typename ProgOptions = AgentProgOptions>
class AAgent : public ProgOptions, public CommonAE {
 public:
  //     virtual AAgent(unsigned int, unsigned int)=0;

  virtual const std::vector<float>& run(float reward, const std::vector<float>&,
                                        bool, bool) = 0;

  virtual void start_episode(const std::vector<float>&) {}

  virtual void save(const std::string&) {}

  std::ostream& display(std::ostream& out, bool display, bool dump) {
    if (display) {
    }

    if (dump) {
    }

    return out;
  }
};
}  // namespace arch

#endif  // AAGENT_H
