#ifndef METROPOLISHASTING_HPP
#define METROPOLISHASTING_HPP

#include <vector>
#include <random>
#include <memory>

#include "Utils.hpp"
#include "Seed.hpp"
#include "RTNorm.hpp"


namespace bib {

template<typename Real>
class Proba {
 public:
  static std::vector<Real>* multidimentionnalGaussian(const std::vector<Real>& centers, Real sigma) {
    std::vector<Real>* gauss = new std::vector<Real>(centers.size());

    for (uint i = 0; i < centers.size(); i++) {
      std::normal_distribution<Real> dist(centers[i], sigma);
      Real number = dist(*bib::Seed::random_engine());
      gauss->at(i) = number;
    }

    return gauss;
  }

  static std::vector<Real>* multidimentionnalTruncatedGaussian(const std::vector<Real>& centers, Real sigma) {
    std::vector<Real>* gauss = new std::vector<Real>(centers.size());

    for (uint i = 0; i < centers.size(); i++)
      gauss->at(i) = rtnorm(-1.0f, 1.0f, centers[i], sigma);

    return gauss;
  }

  static Real truncatedGaussianDensity(Real r, Real mu, const Real sigma, Real a=-1.f, Real b=1.f) {
    //68–95–99.7 rule
    if(mu > b + 3.f*sigma)
      mu = b + 3.f*sigma;
    else if(mu < a - 3.f*sigma)
      mu = a - 3.f*sigma;
    
    const static Real sq2 = 7.071067811865475e-1;            // = 1/sqrt(2)
    const static Real sqpi = 1.772453850905516;              // = sqrt(pi)

    // Scaling
    if(mu!=0 || sigma!=1) {
      a=(a-mu)/sigma;
      b=(b-mu)/sigma;
    }

    Real Z = sqpi *sq2 * sigma * ( erf(b*sq2) - erf(a*sq2) );
    Real p = exp(-pow((r-mu)/sigma,2)/2) / Z;
//     LOG_DEBUG(p << " " << r << " " << mu << " " << Z << " " << a << " " << b <<
//               " " <<erf(b*sq2) << " " << erf(a*sq2));
    return  p;
  }

  static std::vector<Real>* multidimentionnalGaussianWReject(const std::vector<Real>& centers, Real sigma) {
    std::vector<Real>* gauss = new std::vector<Real>(centers.size());

    for (uint i = 0; i < centers.size(); i++) {
      Real number;
      if(centers[i] > 1.0 && centers[i] - 1.0 > 1.5 * sigma) {
        gauss->at(i) = 1.;
        continue;
      } else if(centers[i] < -1.0 && centers[i] + 1.0 < - 1.5 * sigma) {
        gauss->at(i) = -1.;
        continue;
      }

      std::normal_distribution<Real> dist(centers[i], sigma);
      do {
        number = dist(*bib::Seed::random_engine());
      } while (number < -1. || number > 1.);
      gauss->at(i) = number;
    }

    return gauss;
  }

  static Real correlation(const std::vector<Real>& X, const std::vector<Real>& Y) {
    Real Xbar = std::accumulate(X.begin(), X.end(), 0) / X.size();
    Real Ybar = std::accumulate(Y.begin(), Y.end(), 0) / Y.size();

    Real sigmaXY = 0;
    for (uint i = 0; i < X.size() ; i++) {
      sigmaXY += (X[i] - Xbar) * (Y[i] - Ybar);
    }
    sigmaXY /= X.size();

    Real sigmaX = 0;
    for (uint i = 0; i < X.size() ; i++) {
      sigmaX += (X[i] - Xbar) * (X[i] - Xbar);
    }
    sigmaX /= X.size();
    sigmaX = sqrt(sigmaX);

    Real sigmaY = 0;
    for (uint i = 0; i < X.size() ; i++) {
      sigmaY += (Y[i] - Ybar) * (Y[i] - Ybar);
    }
    sigmaY /= Y.size();
    sigmaY = sqrt(sigmaY);

    return sigmaXY / (sigmaX * sigmaY);
  }

  template<class Vector, class F>
  static Real loglikelihood(const Vector& X, F* ptr) {
    Real sum = 0.;
    for(auto it = X.begin(); it != X.end(); ++it)
      sum += log(ptr->density(*it));

    return sum / X.size();
  }
};

template<class T, typename Real>
class MCMC {
 public:
  MCMC(T* _ptr) : ptr(_ptr) {

  }

  std::shared_ptr<std::vector<Real>> oneStep(Real sigma, std::vector<Real>& init, uint step_min=0) {
    Real oldll = log(ptr->eval(init));

    uint step=0;
    while (true) {
      std::shared_ptr<std::vector<Real>> newX(Proba<Real>::multidimentionnalGaussianWReject(init, sigma));
      Real loglike = log(ptr->eval(*newX));
      Real loga = loglike - oldll;

      if (log(bib::Utils::rand01()) < loga) {
        init = *newX;
        oldll = loglike;
        step++;
        if(step > step_min)
          return newX;
      }
    }
  };

  typedef std::shared_ptr<std::vector<Real>> oneLine;

  std::vector<oneLine>* multiStep(uint n, Real sigma, std::vector<Real>& init) {
    Real oldll = log(ptr->eval(init));

    std::vector<oneLine>* accepted = new std::vector<oneLine>(n);

    uint index = 0;
    while (index < n) {
      std::shared_ptr<std::vector<Real>> newX(Proba<Real>::multidimentionnalGaussian(init, sigma));
      Real loglike = log(ptr->eval(*newX));
      Real loga = loglike - oldll;

      if (log(bib::Utils::rand01()) < loga) {
        accepted->at(index++) = newX;
        init = *newX;
        oldll = loglike;
      }
    }

    return accepted;
  }

  std::vector<oneLine>* multiStepReject(uint n, Real sigma, std::vector<Real>& init) {
    Real oldll = log(ptr->eval(init));

    std::vector<oneLine>* accepted = new std::vector<oneLine>(n);

    uint index = 0;
    while (index < n) {
      std::shared_ptr<std::vector<Real>> newX(Proba<Real>::multidimentionnalGaussianWReject(init, sigma));
      Real loglike = log(ptr->eval(*newX));
      Real loga = loglike - oldll;

      if (log(bib::Utils::rand01()) < loga) {
        accepted->at(index++) = newX;
        init = *newX;
        oldll = loglike;
      }
    }

    return accepted;
  }

 protected:
  T* ptr;
};

}

#endif
