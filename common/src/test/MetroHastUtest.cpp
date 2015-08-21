#include "gtest/gtest.h"

#include <math.h>
#include <iostream>
#include "bib/MetropolisHasting.hpp"

struct UnkownDistribution {

  UnkownDistribution() {
    double erf1 = 0.84270079295;
    double analytic_density = (3./2.) * M_PI * erf1 * erf1;
    analytic_density -= (3 * sqrt(M_PI) * erf1)/exp(1);
    density_factor = analytic_density;
  }

  double eval(const std::shared_ptr<std::vector<double>>& x) {
    return eval(*x);
  }

  double eval(const std::vector<double>& x) {
    double x1, x2;
    x1 = x[0];
    x2 = x[1];
    if (x1 > 1)
      x1 = 1;
    else if (x[0] < -1)
      x1 = -1;

    if (x2 > 1)
      x2 = 1;
    else if (x2 < -1)
      x2 = -1;

    return (x1 * x1 + 3 * x2 * x2) * exp(-x1 * x1 - x2 * x2);
  }

  double density(const std::shared_ptr<std::vector<double>>& x) {
    return eval(*x) / density_factor;
  }

  double density(std::vector<double>& x) {
    return eval(x) / density_factor;
  }

 private:
  double density_factor;
};

TEST(MetropolisHasting, OneStepConsistency) {
  UnkownDistribution dist;

  bib::MCMC<UnkownDistribution, double> mcmc(&dist);

  double mean_llw = 0;
  for (uint n = 0; n < 100; n++) {
    std::vector<double> xinit(2, 0);
    xinit[0] = 0;
    xinit[1] = 0;

    uint sample_number = 1000;

    std::ofstream outfile("mcmc.data", std::ios::out);
    double loglike = 0.;
    for (uint i = 0; i < sample_number ; i++) {
      std::shared_ptr<std::vector<double>> point = mcmc.oneStep(0.3, xinit, 8);
      xinit[0] = 0;
      xinit[1] = 0;

      for (const auto & v : *point)
        outfile << v << " ";
      outfile << std::endl;
      loglike += log(dist.density(point));
    }
    loglike /= sample_number;

    outfile.close();
    mean_llw += loglike;
  }

  mean_llw /= 1000.;
  EXPECT_GT(mean_llw, -1. - 0.05);
  LOG_DEBUG("loglikelihood : " << mean_llw);
}

TEST(MetropolisHasting, MultiStepConsistency) {
  UnkownDistribution dist;

  bib::MCMC<UnkownDistribution, double> mcmc(&dist);

  double mean_llw = 0;
  for (uint n = 0; n < 100; n++) {
    std::vector<double> xinit(2, 0);
    xinit[0] = 0;
    xinit[1] = 0;

    std::vector< std::shared_ptr<std::vector<double>> >* points = mcmc.multiStepReject(1000, 0.3, xinit);

    std::ofstream outfile("mcmc2.data", std::ios::out);
    for (auto line = points->begin(); line != points->end(); ++line) {
      std::shared_ptr<std::vector<double>> point = *line;
      for (const auto & v : *point)
        outfile << v << " ";
      outfile << std::endl;
    }

    outfile.close();
    double lg = bib::Proba<double>::loglikelihood<std::vector< std::shared_ptr<std::vector<double> > >, UnkownDistribution>
               (*points, &dist);
    mean_llw += lg;

    delete points;
  }

  mean_llw /= 1000.;
  EXPECT_GT(mean_llw, -1. - 0.05);
  LOG_DEBUG("loglikelihood : " << mean_llw);
}

TEST(MetropolisHasting, MultiStepWithInitConsistency) {
  UnkownDistribution dist;

  bib::MCMC<UnkownDistribution, double> mcmc(&dist);

  double mean_llw = 0;
  for (uint n = 0; n < 100; n++) {
    std::vector<double> xinit(2, 0);
    xinit[0] = 0;
    xinit[1] = 1;

    std::vector< std::shared_ptr<std::vector<double>> >* points = mcmc.multiStepReject(1000, 0.3, xinit);

    std::ofstream outfile("mcmc3.data", std::ios::out);
    for (auto line = points->begin(); line != points->end(); ++line) {
      std::shared_ptr<std::vector<double>> point = *line;
      for (const auto & v : *point)
        outfile << v << " ";
      outfile << std::endl;
    }

    outfile.close();
    double lg = bib::Proba<double>::loglikelihood<std::vector< std::shared_ptr<std::vector<double> > >, UnkownDistribution>
               (*points, &dist);
    mean_llw += lg;

    delete points;
  }

  mean_llw /= 1000;
  EXPECT_GT(mean_llw, -1. - 0.05);
  LOG_DEBUG("loglikelihood : " << mean_llw);

  // octave code
  //clear all; close all;
  //figure; X=load('mcmc.data'); plot3(X(:,1),X(:,2), '.'); axis([-1 1 -1 1]);
  //figure; Y=load('mcmc2.data'); plot3(Y(:,1),Y(:,2), '.'); axis([-1 1 -1 1]);
  //figure; Z=load('mcmc3.data'); plot3(Z(:,1),Z(:,2), '.'); axis([-1 1 -1 1]);

}
