#ifndef ORNSTEINUHLENBECK_HPP
#define ORNSTEINUHLENBECK_HPP

#include <vector>

namespace bib {
  
template<typename Real>
class OrnsteinUhlenbeckNoise {
public:
  OrnsteinUhlenbeckNoise(uint action_size, Real sigma_=.2, Real theta_=.15, Real dt_=0.01) : x_t (action_size, 0) {
      sigma = sigma_;
      theta = theta_;
      dt = dt_;
  }

  void reset() {
    for(uint i=0;i<x_t.size();i++)
      x_t[i] = 0;
  }

  void step(std::vector<Real>& mu) {
    std::normal_distribution<Real> dist(0, 1.);
    
    for (uint i = 0; i < mu.size(); i++) {
      Real normal = dist(*bib::Seed::random_engine());
      Real x_tplus = x_t[i] - theta * dt * x_t[i] + sigma * sqrt(dt) * normal;
      mu[i] += x_tplus;
      
      if(mu[i] < (Real) -1.)
        mu[i] = (Real) -1.;
      else if (mu[i] > (Real) 1.)
        mu[i] = (Real) 1.;
      
      x_t[i] = x_tplus;
    }
  }

private:
  std::vector<Real> x_t;
  Real sigma;
  Real theta;
  Real dt;
};
  
}

#endif
