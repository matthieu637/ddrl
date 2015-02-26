#include "bib/Prober.hpp"
#include <algorithm>
#include "bib/Logger.hpp"

namespace bib {

void Prober::probe(float m) {
  if (prob_init) {
    min_probe = std::min(min_probe, m);
    max_probe = std::max(max_probe, m);
  } else {
    min_probe = m;
    max_probe = m;
    prob_init = true;
  }

  prob_step++;

  if (prob_step % 100 == 0) {
    LOG_DEBUG("min : " << min_probe << " max : " << max_probe);
  }
}
}
