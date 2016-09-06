#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include <doublefann.h>
#include <fann_train.h>
#include <fann.h>
#include <boost/filesystem.hpp>

#include "MLP.hpp"
#include <bib/Utils.hpp>
#include <bib/MetropolisHasting.hpp>
#include <bib/Combinaison.hpp>

#define NB_SOL_OPTIMIZATION 25

double OR(double x1, double x2) {
  if (x1 == 1.f || x2 == 1.f)
    return 1.f;
  return -1.f;
}

double AND(double x1, double x2) {
  if (x1 == 1.f && x2 == 1.f)
    return 1.f;
  return -1.f;
}

double LECUNIZATION(double x){ 
  if(x == 1.f) 
    return 1.5f;
  else
    return -1.5f;
}

double sech(double x) {
  return 2. / (exp(x) + exp(-x));
}

TEST(MLP, ConsistentActivationFunction) {
  MLP nn(1, {1}, 0, 0.01f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 1.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  std::vector<double> sens(0);
  std::vector<double> ac(1);
  ac[0] = 1;

  double lambda = 0.5;
  fann_set_activation_steepness_output(nn.getNeuralNet(), lambda);
  fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(lambda * 1.) / 2.);

  lambda = 0.8;
  fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(lambda * 1.) / 2.);

  ac[0] = -1;
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(lambda * -1.) / 2.);

  lambda = 0.5;
  fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), tanh(lambda * -1.) / 2.);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 0.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 5.f);

  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), 5.f / 2.f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 0.2f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.4f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 0.6f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.8f);

  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), (tanh(lambda * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) / 2.f);

  fann_set_activation_steepness_output(nn.getNeuralNet(), 1.f);
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac),  tanh(lambda * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f);
}


TEST(MLP, LearnOpposite) {
  MLP nn(1, {1}, 0, 0.5f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, -1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);

  fann_set_weight(nn.getNeuralNet(), 2, 4, 1. / tanh(0.5 * 1.));
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(0);
    std::vector<double> ac(1);
    ac[0] = x1;

    EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), out);
  }

  // Learn
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(0);
    std::vector<double> ac(1);
    ac[0] = x1;

    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(0);
    std::vector<double> ac(1);
    ac[0] = x1;

    EXPECT_EQ(nn.computeOutVF(sens, ac), out);
  }
}


TEST(MLP, LearnAndOr) {
  MLP nn(3, {10}, 2, 0.5f);

  // Learn
  for (uint n = 0; n < 10000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.02);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.02);
  }
}


TEST(MLP, MLPCheckOuput) {
  MLP nn(1, {1}, 2, 0.5f);
  std::vector<double> sens(0);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 1.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  uint ndim = 1;
  passdata d = {nn.getNeuralNet(), sens};
  ColumnVector ac(ndim);
  ac(1) = -1.f;
  double fx = -1;
  int result = 0;
  ColumnVector gx(ndim);
  SymmetricMatrix Hx(ndim);
  hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);

  EXPECT_DOUBLE_EQ(fx, tanh(0.5 * 1.));

  fann_set_weight(nn.getNeuralNet(), 0, 2, 0.2f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.4f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 0.6f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.8f);

  hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);

  EXPECT_DOUBLE_EQ(-fx, tanh(0.5 * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f);
}

TEST(MLP, MLPCheckDerivative) {
  MLP nn(1, {1}, 2, 0.5f);
  std::vector<double> sens(0);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 1.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  uint ndim = 1;
  passdata d = {nn.getNeuralNet(), sens};
  ColumnVector ac(ndim);
  double a = -1;
  ac(1) = a;
  double fx = -1;
  int result = 0;

  ColumnVector gx(ndim);

  SymmetricMatrix Hx(ndim);

  double h = 1e-10;
  ac(1) = a - h;
  hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
  double fx_base = fx;
  ac(1) = a + h;
  hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
  double fx_h = fx;
  double derivative1 = (fx_h - fx_base) / (2.0 * h);

  ac(1) = a;
  hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
  double derivative2 = (fx_h - fx) / h;

  hs65(NLPGradient, ndim, ac, fx, gx, Hx, result, &d);

  EXPECT_GT(derivative1, gx(1) - 1e-5);
  EXPECT_LT(derivative1, gx(1) + 1e-5);

  EXPECT_GT(derivative2, gx(1) - 1e-5);
  EXPECT_LT(derivative2, gx(1) + 1e-5);
}

TEST(MLP, MLPCheckAnalyticscFuncDerHess) {
  MLP nn(1, {1}, 0, 0.5f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, -1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);

  fann_set_weight(nn.getNeuralNet(), 2, 4, 1. / tanh(0.5 * 1.));
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  uint ndim = 1;
  std::vector<double> sens(0);
  passdata d = {nn.getNeuralNet(), sens};
  ColumnVector ac(ndim);
  double a = -1;
  ac(1) = a;
  double fx = -1;
  int result = 0;
  ColumnVector gx(ndim);

  SymmetricMatrix Hx(ndim);

  double C = 1. / (2. * tanh(0.5)) ;

  //exact computation
  for (a = -1; a <= 1.0; a += 1e-2) {
    ac(1) = a;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    EXPECT_GT(- fx, tanh(-0.5 * a) / tanh(0.5)  - 1e-7);  //opposite taking to maximaze
    EXPECT_LT(- fx, tanh(-0.5 * a) / tanh(0.5)  + 1e-7);

    hs65(NLPGradient, ndim, ac, fx, gx, Hx, result, &d);
    EXPECT_GT(- gx(1), - C * sech(0.5 * a) * sech(0.5 * a) - 1e-7);
    EXPECT_LT(- gx(1), - C * sech(0.5 * a) * sech(0.5 * a) + 1e-7);

    hs65(NLPHessian, ndim, ac, fx, gx, Hx, result, &d);
    EXPECT_GT(- Hx(1, 1), C * tanh(0.5 * a) * sech(0.5 * a) * sech(0.5 * a) - 1e-7);
    EXPECT_LT(- Hx(1, 1), C * tanh(0.5 * a) * sech(0.5 * a) * sech(0.5 * a) + 1e-7);
  }
}

TEST(MLP, MLPCheckDerivativeHard) {
  MLP nn(4, {10}, 3, 0.5f);
  std::vector<double> sens(3);
  sens[0] = bib::Utils::randin(-1, 1);
  sens[1] = bib::Utils::randin(-1, 1);
  sens[2] = bib::Utils::randin(-1, 1);

  uint ndim = 1;
  passdata d = {nn.getNeuralNet(), sens};
  ColumnVector ac(ndim);
  ColumnVector gx(ndim);
  SymmetricMatrix Hx(ndim);
  int result = 0;
  double fx = -1;

  for (double a = -1; a <= 1.0; a += 1e-2) {
    ac(1) = a;

    double h = 1e-10;
    ac(1) = a - h;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double fx_base = fx;
    ac(1) = a + h;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double fx_h = fx;
    double derivative1 = (fx_h - fx_base) / (2.0 * h);

    ac(1) = a;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double derivative2 = (fx_h - fx) / h;

    hs65(NLPGradient, ndim, ac, fx, gx, Hx, result, &d);

    EXPECT_GT(derivative1, gx(1) - 1e-5);
    EXPECT_LT(derivative1, gx(1) + 1e-5);

    EXPECT_GT(derivative2, gx(1) - 1e-5);
    EXPECT_LT(derivative2, gx(1) + 1e-5);
  }
}

TEST(MLP, MLPCheckHessian) {
  MLP nn(4, {10}, 3, 0.5f);
  std::vector<double> sens(3);
  sens[0] = bib::Utils::randin(-1, 1);
  sens[1] = bib::Utils::randin(-1, 1);
  sens[2] = bib::Utils::randin(-1, 1);

  uint ndim = 1;
  passdata d = {nn.getNeuralNet(), sens};
  ColumnVector ac(ndim);
  ColumnVector gx(ndim);
  SymmetricMatrix Hx(ndim);
  int result = 0;
  double fx = -1;

  for (double a = -1; a <= 1.0; a += 1e-2) {
    ac(1) = a;

    double h = 1e-5;
    ac(1) = a - h;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double fx_base = fx;
    ac(1) = a + h;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double fx_h = fx;

    ac(1) = a;
    hs65(NLPFunction, ndim, ac, fx, gx, Hx, result, &d);
    double snd_derivative = (fx_h - 2.0 * fx + fx_base) / (h * h);

    hs65(NLPHessian, ndim, ac, fx, gx, Hx, result, &d);

    EXPECT_GT(snd_derivative, Hx(1, 1) - 1e-5);
    EXPECT_LT(snd_derivative, Hx(1, 1) + 1e-5);
  }
}

TEST(MLP, OptimizeOpposite) {
  MLP nn(1, {1}, 0, 0.5f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, -1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);

  fann_set_weight(nn.getNeuralNet(), 2, 4, 1. / tanh(0.5 * 1.));
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

  // Test
  std::vector<double> sens(0);
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> ac(1);
    ac[0] = x1;

    EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), out);
  }

  std::vector<double>* ac = nn.optimized(sens);
  EXPECT_GT(ac->at(0), -1. - 0.01);
  EXPECT_LT(ac->at(0), -1. + 0.01);
  delete ac;

  fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
  ac = nn.optimized(sens);
  EXPECT_GT(ac->at(0), 1. - 0.01);
  EXPECT_LT(ac->at(0), 1. + 0.01);
  delete ac;

  // Learn
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = x1 == 1.f ? -1.f : 1.f;

    std::vector<double> sens(0);
    std::vector<double> acc(1);
    acc[0] = x1;

    nn.learn(sens, acc, out);
  }

  ac = nn.optimized(sens);
  EXPECT_GT(ac->at(0), -1. - 0.01);
  EXPECT_LT(ac->at(0), -1. + 0.01);

//     LOG_DEBUG(nn.computeOut(sens, *ac));
  ac->at(0) = -1;
//     LOG_DEBUG(nn.computeOut(sens, *ac));
  delete ac;
}


TEST(MLP, OptimizeAndOr) {
  MLP nn(3, {10}, 2, 0.5f);

  // Learn
  for (uint n = 0; n < 10000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    nn.learn(sens, ac, out);
  }



  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;

    std::vector<double>* ac = nn.optimized(sens);
    double his_sol = nn.computeOutVF(sens, *ac);

    ac->at(0) = 1.f;
    double my_sol = nn.computeOutVF(sens, *ac);

    EXPECT_GE(his_sol, my_sol - 0.01);
    delete ac;
  }
}


TEST(MLP, OptimizeMultiDim) {
  MLP nn(3, {20}, 0, 0.5f);

  // Learn
  for (uint n = 0; n < 10000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(0);
    std::vector<double> ac(3);
    ac[0] = x1;
    ac[1] = x2;
    ac[2] = x3;

    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    std::vector<double> sens(0);
    std::vector<double>* ac = nn.optimized(sens);
    double his_sol = nn.computeOutVF(sens, *ac);
    double my_sol = 1.;

    EXPECT_GE(his_sol, my_sol - 0.01);
    delete ac;
  }
}

TEST(MLP, OptimizeNonExtremum) {
  //learn x^2 - x
  //root min ~0.474
  //root max -1

  double sensv = 1. ;
  for (uint n = 0; n < 1 ; n++) {
    MLP nn(1, {3}, 0, 0.2f);
    std::vector<double> sens(0);
    std::vector<double> ac(1);

    // Learn
    for (uint n = 0; n < 100000 ; n++) {
      double x1 = (bib::Utils::rand01() * 2.) - 1;

      double out = x1 * x1 - x1;
      ac[0] = x1;

      nn.learn(sens, ac, sensv * out);
    }

    //Test learned
    for (uint n = 0; n < 1000 ; n++) {
      double x1 = (bib::Utils::rand01() * 2.) - 1;

      double out = x1 * x1 - x1;
      ac[0] = x1;

      double myout = nn.computeOutVF(sens, ac);
      EXPECT_GT(myout, sensv * out - 0.1);
      EXPECT_LT(myout, sensv * out + 0.1);
      LOG_FILE("OptimizeNonExtremum.data", x1 << " " << myout << " " << out);
//     close all; X=load('OptimizeNonExtremum.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');
    }

    std::vector<double>* acopt = nn.optimized(sens, {}, NB_SOL_OPTIMIZATION);
    if (sensv == -1.) {
      EXPECT_GT(acopt->at(0), 0.46);
      EXPECT_LT(acopt->at(0), 0.48);
    } else {
      EXPECT_LT(acopt->at(0), -0.99);
      EXPECT_GT(acopt->at(0), -1.01);
    }
    delete acopt;
    sensv *= -1;
  }
}


TEST(MLP, Optimize2LocalMaxima) {
  //learn - cos(5.*x1)/2.

  MLP nn(1, {10}, 0, 0.1f);
  std::vector<double> sens(0);
  std::vector<double> ac(1);

  fann_set_training_algorithm(nn.getNeuralNet(), FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
  fann_set_train_error_function(nn.getNeuralNet(), FANN_ERRORFUNC_TANH);

  struct fann_train_data* data = fann_create_train(2500, 1, 1);

  // Learn
  for (uint n = 0; n < 2500 ; n++) {
    double x1 = (bib::Utils::rand01() * 2.) - 1;

    double out = - cos(5.*x1) / 2.;
    ac[0] = x1;

    data->input[n][0] = x1;
    data->output[n][0] = out;
  }

  fann_train_on_data(nn.getNeuralNet(), data, 3000, 0, 0.0005);
  fann_destroy_train(data);

  //Test learned
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = (bib::Utils::rand01() * 2.) - 1;

    double out = - cos(5.*x1) / 2.;
    ac[0] = x1;

    double myout = nn.computeOutVF(sens, ac);
    LOG_FILE("Optimize2LocalMaxima.data", x1 << " " << myout << " " << out);
//     close all; X=load('Optimize2LocalMaxima.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');
  }

  std::vector<double>* acopt = nn.optimized(sens, {}, NB_SOL_OPTIMIZATION);
//     ac[0] = acopt->at(0);
//     double my_sol = nn.computeOut(sens, ac);
//     ac[0] = M_PI/5;
//     double best_sol = nn.computeOut(sens, ac);
//     LOG_DEBUG("my sol : " << my_sol << " | best : " << best_sol << " (" << acopt->at(0) << " vs " << M_PI/5 << ") ");

  double precision = 0.1;
  if (acopt->at(0) > 0) {
    EXPECT_GT(acopt->at(0), M_PI / 5. - precision);
    EXPECT_LT(acopt->at(0), M_PI / 5. + precision);
  } else {
    EXPECT_GT(acopt->at(0), -M_PI / 5. - precision);
    EXPECT_LT(acopt->at(0), -M_PI / 5 + precision);
  }

  delete acopt;
}


TEST(MLP, OptimizeTrapInEvilLocalOptimal) {
  //learn sin(-3*x*x*x+2*x*x+x)
  //local max ~0.6
  //global max -0.7

  MLP nn(1, {12}, 0, 0.1f);
  std::vector<double> sens(0);
  std::vector<double> ac(1);

  fann_set_training_algorithm(nn.getNeuralNet(), FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
  fann_set_train_error_function(nn.getNeuralNet(), FANN_ERRORFUNC_TANH);

  struct fann_train_data* data = fann_create_train(2500, 1, 1);

  // Learn
  for (uint n = 0; n < 2500 ; n++) {
    double x1 = (bib::Utils::rand01() * 2.) - 1;

    double out = sin(-3 * x1 * x1 * x1 + 2 * x1 * x1 + x1);
    ac[0] = x1;

    data->input[n][0] = x1;
    data->output[n][0] = out;
  }

  fann_train_on_data(nn.getNeuralNet(), data, 10000, 0, 0.0005);
  fann_destroy_train(data);

  //Test learned
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = (bib::Utils::rand01() * 2.) - 1;

    double out = sin(-3 * x1 * x1 * x1 + 2 * x1 * x1 + x1);
    ac[0] = x1;

    double myout = nn.computeOutVF(sens, ac);
    LOG_FILE("OptimizeTrapInEvilLocalOptimal.data", x1 << " " << myout << " " << out);
//     close all; X=load('OptimizeTrapInEvilLocalOptimal.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');
  }

  std::vector<double>* acopt = nn.optimized(sens, {}, NB_SOL_OPTIMIZATION+100);
//     LOG_DEBUG(acopt->at(0));

  double precision = 0.1;

  if(acopt->at(0) < -0.7 - precision || acopt->at(0) > -0.7 + precision) {
    LOG_DEBUG(nn.computeOutVF(sens, *acopt));
    ac[0] = -0.7;
    LOG_DEBUG(nn.computeOutVF(sens, ac));
  }

  EXPECT_GT(acopt->at(0), -0.7 - precision);
  EXPECT_LT(acopt->at(0), -0.7 + precision);

  delete acopt;
}



TEST(MLP, OptimizePlateau) {
  //learn sin(-3*x*x*x+2*x*x+x)
  //local max ~0.6
  //global max -0.7

  MLP nn(1, {2}, 0, 0.1f);
  std::vector<double> sens(0);
  std::vector<double> ac(1);

  fann_set_training_algorithm(nn.getNeuralNet(), FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
  fann_set_train_error_function(nn.getNeuralNet(), FANN_ERRORFUNC_TANH);

  struct fann_train_data* data = fann_create_train(5000, 1, 1);

  // Learn
  for (uint n = 0; n < 5000 ; n++) {
    double x1 = n < 2000 ? (bib::Utils::rand01() * 2.) - 1 : bib::Utils::randin(0, 0.5);

    double out = x1 >= 0.2 ? 1. : 0.2;
    ac[0] = x1;

    data->input[n][0] = x1;
    data->output[n][0] = out;
  }

  fann_train_on_data(nn.getNeuralNet(), data, 10000, 0, 0.00001);
  fann_destroy_train(data);

  //Test learned
  for (uint n = 0; n < 1000 ; n++) {
    double x1 = (bib::Utils::rand01() * 2.) - 1;

    double out = x1 >= 0.2 ? 1. : 0.2;
    ac[0] = x1;

    double myout = nn.computeOutVF(sens, ac);
    LOG_FILE("OptimizePlateau.data", x1 << " " << myout << " " << out);
//     close all; X=load('OptimizePlateau.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');
  }

  std::vector<double>* acopt = nn.optimized(sens, {}, NB_SOL_OPTIMIZATION);
//     LOG_DEBUG(acopt->at(0));

  double precision = 0.01;

  EXPECT_GT(acopt->at(0), 0.2 - precision);

  delete acopt;
}

TEST(MLP, ConsistentActivationFunctionLecun) {
  MLP nn(1, {1}, 0, 0.01f, true);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 1.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);
  
  std::vector<double> sens(0);
  std::vector<double> ac(1);
  ac[0] = 1.d;

  double lambda = atanh(1.d/sqrt(3.d));
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), sqrt(3.d)*tanh(lambda * 1.d));

  ac[0] = 1.5d;
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), sqrt(3.d)*tanh(lambda * 1.5d));//1.31
  
  fann_set_weight(nn.getNeuralNet(), 0, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 0.f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 5.f);

  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), 5.f);

  fann_set_weight(nn.getNeuralNet(), 0, 2, 0.2f);
  fann_set_weight(nn.getNeuralNet(), 1, 2, 0.4f);
  fann_set_weight(nn.getNeuralNet(), 2, 4, 0.6f);
  fann_set_weight(nn.getNeuralNet(), 3, 4, 0.8f);

  ac[0] = -1.d;
  EXPECT_DOUBLE_EQ(nn.computeOutVF(sens, ac), (sqrt(3.d)*tanh(lambda * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f));
}

TEST(MLP, LearnAndOrLecun) {
  MLP nn(3, {10}, 2, 0.05f, true);

  // Learn
  for (uint n = 0; n < 10000 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;
    
    nn.learn(sens, ac, out);
  }

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.02);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.02);
  }
}

TEST(MLP, LearnAndOrLecun2) {
  MLP nn(3, {5}, 1, true);

  struct fann_train_data* data = fann_create_train(2*2*2, 3, 1);
   
  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->input[n][1]= x[1];
    data->input[n][2]= x[2];
    
    double out = AND(OR(x[0], x[1]), x[2]);
    data->output[n][0]= out;
    n++;
  };
    
  bib::Combinaison::continuous<>(iter, 3, -1, 1, 1);
  
  nn.learn_stoch(data, 20000, 0, 0.00000001);
  
  fann_destroy_train(data);

  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::randBool() ? 1.f : -1.f;
    double x2 = bib::Utils::randBool() ? 1.f : -1.f;
    double x3 = bib::Utils::randBool() ? 1.f : -1.f;

    double out = AND(OR(x1, x2), x3);

    std::vector<double> sens(2);
    sens[0] = x1;
    sens[1] = x2;
    std::vector<double> ac(1);
    ac[0] = x3;

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.02);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.02);
    
    std::vector<double> in(3);
    in[0] = x1;
    in[1] = x2;
    in[2] = x3;
    std::vector<double>* outnn = nn.computeOut(in);
    
    EXPECT_GT(outnn->at(0), out - 0.05);
    EXPECT_LT(outnn->at(0), out + 0.05);
    
    delete outnn;
  }
}


// 1 0.2
// 3 0.3
// 5 0.3
// 
// Result:  y = 0.025 x + 1.916666667·10-1
// 
// 2.166666667·10-1 		 1.666666667·10-2 
// 2.666666667·10-1 		 3.333333333·10-2 
// 3.166666667·10-1 		 1.666666667·10-2 
TEST(MLP, LearnLabelWeight) {
  fann* lin_nn =fann_create_standard(2, 1, 1);
  fann_set_activation_function_output(lin_nn, FANN_LINEAR);
  fann_set_learning_momentum(lin_nn, 0.);
  fann_set_training_algorithm(lin_nn, FANN_TRAIN_RPROP);

  struct fann_train_data* data = fann_create_train(3, 1, 1);
  fann_type lw [3];
  
  uint n = 0;
  data->input[n][0]= 1.f;
  data->output[n][0]= 2.f/10.f;
  lw[n] = 1.f;
  n++;
  data->input[n][0]= 3.f;
  data->output[n][0]= 3.f/10.f;
  lw[n] = 0.5f;
  n++;
  data->input[n][0]= 5.f;
  data->output[n][0]= 3.f/10.;
  lw[n] = 1.f;
  n++;
  
  for(int i=0;i<1000; i++)
    fann_train_epoch(lin_nn, data);
  
  fann_type out_no_lw[3];
  fann_type * out;
  n=0;
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.216 - 0.002);
  EXPECT_LT(out[0], 0.216 + 0.002);
  out_no_lw[n]=out[0];
  n++;

  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.266 - 0.002);
  EXPECT_LT(out[0], 0.266 + 0.002);
  out_no_lw[n]=out[0];
  n++;
  
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.316 - 0.002);
  EXPECT_LT(out[0], 0.316 + 0.002);
  out_no_lw[n]=out[0];
  
  for(int i=0;i<1000; i++)
    fann_train_epoch_lw(lin_nn, data, lw);
  
  n=0;
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.2);
  EXPECT_LT(out[0], out_no_lw[n]);
  out_no_lw[n]=out[0];
  n++;

  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.25);
  EXPECT_LT(out[0], out_no_lw[n]);
  out_no_lw[n]=out[0];
  n++;
  
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.3);
  EXPECT_LT(out[0], out_no_lw[n]);
  out_no_lw[n]=out[0];
  
  lw[1] = 0.05;
  for(int i=0;i<1000; i++)
    fann_train_epoch_lw(lin_nn, data, lw);
    
  n=0;
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.2);
  EXPECT_LT(out[0], out_no_lw[n]);
  n++;

  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.25);
  EXPECT_LT(out[0], out_no_lw[n]);
  n++;
  
  out = fann_run(lin_nn, data->input[n]);
  EXPECT_GT(out[0], 0.3);
  EXPECT_LT(out[0], out_no_lw[n]);
  
  fann_destroy_train(data);
}

TEST(MLP, LearnNonLinearLabelWeight) {
  MLP nn(1, {4}, 1, true);

  struct fann_train_data* data = fann_create_train(11*2, 1, 1);
  fann_type label_weight[11*2];
  
  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->output[n][0]= x[0];
    label_weight[n] = 1.f;
    n++;
  };
  
  auto iter2 = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->output[n][0]= x[0]*x[0];
    label_weight[n] = 0.2f;
    n++;
  };
  
  bib::Combinaison::continuous<>(iter, 1, 0, 1, 10);
  bib::Combinaison::continuous<>(iter2, 1, 0, 1, 10);
  
  nn.learn_stoch(data, 20000, 0, 0.00000001);
  
  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01();

    double out = (x1 + x1*x1)/2.;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);
    
    double old_out1 = x1;
    double old_out2 = x1*x1;

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.15);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.15);
    
    if(x1 > 0.15 && x1 < 0.85){
      EXPECT_GE(fabs(nn.computeOutVF(sens, ac) - old_out1), fabs(nn.computeOutVF(sens, ac) - out));
      EXPECT_GE(fabs(nn.computeOutVF(sens, ac) - old_out2), fabs(nn.computeOutVF(sens, ac) - out));
    }
      
  }
  
  nn.learn_stoch_lw(data, label_weight, 20000, 0, 0.00000001);
  
  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01();

    double old_out = (x1 + x1*x1)/2.;
    double out = (x1 + 0.2*x1*x1)/1.2;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.1);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.1);
    
    if(x1 > 0.15 && x1 < 0.85)
      EXPECT_GE(fabs(nn.computeOutVF(sens, ac) - old_out), fabs(nn.computeOutVF(sens, ac) - out));
  }
  
  fann_destroy_train(data);
}


TEST(MLP, LearnNonLinearLabelWeightWithNullImportance) {
  MLP nn(1, {4}, 1, true);

  struct fann_train_data* data = fann_create_train(11*2, 1, 1);
  fann_type label_weight[11*2];
  
  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->output[n][0]= x[0];
    label_weight[n] = 0.0f;
    n++;
  };
  
  auto iter2 = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->output[n][0]= -x[0];
    label_weight[n] = 0.8f;
    n++;
  };
  
  bib::Combinaison::continuous<>(iter, 1, 0, 1, 10);
  bib::Combinaison::continuous<>(iter2, 1, 0, 1, 10);
  
  nn.learn_stoch_lw(data, label_weight, 20000, 0, 0.00000001);
  
  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01();

    double out = (0.0*x1 - 0.8*x1)/(0.0+0.8);

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.15);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.15);
  }
  
  fann_destroy_train(data);
}


double derivative(double* s, double* a, int, void*){
    return -(2*a[0] -s[0]*s[0]);
}

// try to learn a function pi that maximize another one : f(s, pi(a)) = a^2-(s^2)*a
TEST(MLP, OptimizeNNTroughGradient) {
  MLP nn(1, {4}, 1, 0.0, true);

  struct fann_train_data* data = fann_create_train(200, 1, 1);
  
  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
//     data->output[n][0]= -x[0]*x[0];
    data->output[n][0]= x[0];//don't care
    n++;
  };
  
  bib::Combinaison::continuous<>(iter, 1, -1, 1, 200);
  
  for(int i=0;i<10000; i++)
    fann_train_epoch_irpropm_gradient(nn.getNeuralNet(), data, derivative, nullptr);
  
  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01()*2 -1;

    double out = (x1 * x1)/2;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(0);

    EXPECT_GT(nn.computeOutVF(sens, ac), out - 0.1);
    EXPECT_LT(nn.computeOutVF(sens, ac), out + 0.1);
    
    LOG_FILE("OptimizeNNTroughGradient.data", x1 << " " << nn.computeOutVF(sens, ac) << " " << out);
  }
  
  fann_destroy_train(data);
}


class my_weights {
  typedef struct fann_connection sfn;

 public:
  my_weights(NN neural_net, const double* sensors, const double* x, uint _m, uint _n) : m(_m), n(_n) {
    _lambda = fann_get_activation_steepness(neural_net, 1, 0);

    unsigned int number_connection = fann_get_total_connections(neural_net);
    connections = reinterpret_cast<sfn*>(calloc(number_connection, sizeof(sfn)));

    fann_get_connection_array(neural_net, connections);

    uint number_layer = fann_get_num_layers(neural_net);
    layers = reinterpret_cast<uint*>(calloc(number_layer, sizeof(sfn)));

    fann_get_layer_array(neural_net, layers);
    ASSERT(number_layer == 3, number_layer);

    for (uint i = 0; i < h() * (m + n + 1); i++) {
      _ASSERT_EQ(connections[i].from_neuron, i % (m + n + 1));
      _ASSERT_EQ(connections[i].to_neuron, (m + n + 1) + i / (m + n + 1));
    }

    Ci.clear();
    Ci.resize(h());
    for (uint i = 0; i < h(); i++) {
      Ci[i] = 0;
      for (uint j = 0; j < m; j++)
        Ci[i] += sensors[j] * w(j, i);

      Ci[i] += connections[ i * (m + n + 1) + m + n].weight;
      _ASSERT_EQ(connections[ i * (m + n + 1) + m + n].from_neuron, m + n);
      _ASSERT_EQ(connections[ i * (m + n + 1) + m + n].to_neuron, m + n + 1 + i);
    }

    Di.clear();
    Di.resize(h());
    for (uint i = 0; i < h(); i++) {
      Di[i] = Ci[i];
      for (uint j = 0; j < n; j++)
        Di[i] += x[j] * w(m + j, i);
    }

    ASSERT(number_connection == h() * (m + n + 1) + (h() + 1), "");
  }

  ~my_weights() {
    free(connections);
    free(layers);
  }

  double v(uint i) const {
    sfn conn = connections[ h() * (m + n + 1) + i];

    _ASSERT_EQS(conn.from_neuron, (m + n + 1) + i, " i: " << i  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m + n + 1) + (h() + 1), " i: " << i  << " m " << m << " n " << n << " h " << h());
    return conn.weight;
  }

  double w(uint j, uint i) const {
    sfn conn = connections[ i * (m + n + 1) + j];

    _ASSERT_EQS(conn.from_neuron, j, " i: " << i << " j : " << j  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m + n + 1) + i, " i: " << i << " j : " << j  << " m " << m << " n " << n << " h " << h());
    return conn.weight;
  }

  uint h() const {
    return layers[1];
  }

  double C(uint i) const {
    return Ci[i];
  }

  double D(uint i) const {
    return Di[i];
  }

  double lambda() const {
    return _lambda;
  }

 private :
  sfn* connections;
  uint* layers;
  std::vector<double> Ci;
  std::vector<double> Di;
  uint m, n;
  double _lambda;
};

#define activation_function(x, lambda) tanh(lambda * x)

double derivative2(double* input, double *neuron_value, int, void* data){
    MLP* nn = (MLP*)data;
    my_weights _w(nn->getNeuralNet(), input, neuron_value, 1, 1);
    
    std::vector<double> gx(1);
    for (uint j = 0; j < 1; j++) {
      gx[j] = 0;
      for (uint i = 0; i < _w.h() ; i++) {
        double der = activation_function(_w.D(i), _w.lambda());
        gx[j] = gx[j] + _w.v(i) * _w.w(1 + j, i) * _w.lambda() * (1.0 - der * der);
      }
    }

//     LOG_DEBUG((-(2*neuron_value[0] -input[0]*input[0])) << " " << gx[0]);
    return gx[0];
}

TEST(MLP, OptimizeNNTroughGradientOfAnotherNN) {
  MLP nn(2, {50}, 1, 0.0);
  MLP actor(1, {4}, 1);

  struct fann_train_data* data = fann_create_train(200*200, 2, 1);
  
  uint n = 0;
  auto iter = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->input[n][1]= x[1];

    data->output[n][0]= -(x[1]*x[1]-(x[0]*x[0])*x[1]);
    n++;
  };
  
  bib::Combinaison::continuous<>(iter, 2, -1, 1, 200);
  
  nn.learn_stoch(data, 500, 100, 0.0000001,200);
  fann_destroy_train(data);
  
  data = fann_create_train(2000, 1, 1);
  n = 0;
  auto iter2 = [&](const std::vector<double>& x) {
    data->input[n][0]= x[0];
    data->output[n][0]= x[0];//don't care
    n++;
  };
  bib::Combinaison::continuous<>(iter2, 1, -1, 1, 2000);
  
  //fann_type *error_begin = nn.getNeuralNet()->train_errors;
  
  for(int i=0;i<1000; i++)
     fann_train_epoch_irpropm_gradient(actor.getNeuralNet(), data, derivative2, &nn);
  
  // Test
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01()*2 -1;

    double out = (x1 * x1)/2;
    double qout = -(out*out-(x1*x1)*out);

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(1);
    std::vector<double> ac_empty(0);
    ac[0]= out;
    
    EXPECT_GT(nn.computeOutVF(sens, ac), qout - 0.15);
    EXPECT_LT(nn.computeOutVF(sens, ac), qout + 0.15);

    EXPECT_GT(actor.computeOutVF(sens, ac_empty), out - 0.2);
    EXPECT_LT(actor.computeOutVF(sens, ac_empty), out + 0.2);
    
    LOG_FILE("OptimizeNNTroughGradientOfAnotherNN.data", x1 << " " << actor.computeOutVF(sens, ac_empty) << " " << out);
  }
  
  fann_destroy_train(data);
}

TEST(MLP, OptimizeNNTroughGradientOfAnotherNNFann) {
  MLP nn(2, {50}, 1, 0.0, true);
//   MLP nn(2, 1, 1, 0.0);
  MLP actor(1, {8}, 1);
  
  MLP actor2(actor);
  fann_set_activation_function_output(actor.getNeuralNet(), FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(actor2.getNeuralNet(), FANN_LINEAR);

  struct fann_train_data* data;
  if ( !boost::filesystem::exists( "OptimizeNNTroughGradientOfAnotherNNFann.cache.data" ) ){
    data = fann_create_train(200*200, 2, 1);
    
    uint n = 0;
    auto iter = [&](const std::vector<double>& x) {
      data->input[n][0]= x[0];
      data->input[n][1]= x[1];

      data->output[n][0]= -(x[1]*x[1]-(x[0]*x[0])*x[1]);
      n++;
    };
    
    bib::Combinaison::continuous<>(iter, 2, -1, 1, 200);
    
    
    nn.learn_stoch(data, 5800, 100, 0.0000001,500);
    //nn.learn(data, 300, 0);
    
    fann_destroy_train(data);
    nn.save("OptimizeNNTroughGradientOfAnotherNNFann.cache.data");
  } else 
    nn.load("OptimizeNNTroughGradientOfAnotherNNFann.cache.data");
  

  
  datann_derivative d = {&nn, 1, 1};
  
  for(int i=0;i<5000; i++){
     data = fann_create_train(4000, 1, 1);
     for (uint n = 0; n < 4000 ; n++){
       data->input[n][0]= bib::Utils::rand01()*2-1.;
       data->output[n][0]= 0;
     }
    
    
     fann_train_epoch_irpropm_gradient(actor.getNeuralNet(), data, derivative_nn, &d);

     fann_train_epoch_irpropm_gradient(actor2.getNeuralNet(), data, derivative_nn_inverting, &d);
     if(i%200 == 0)
      LOG_DEBUG(actor.weight_l1_norm() << " " << actor2.weight_l1_norm());
     
     fann_destroy_train(data);
  }
  
  // Test
  double ac1_error = 0;
  double ac2_error = 0;
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01()*2 -1;

    double out = (x1 * x1)/2;
    //out = x1 > 0 ? 0 : -1.f;
    double qout = out*out-(x1*x1)*out;
    qout = - qout;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(1);
    std::vector<double> ac_empty(0);
    ac[0]= out;
    
    EXPECT_GT(nn.computeOutVF(sens, ac), qout - 0.2);
    EXPECT_LT(nn.computeOutVF(sens, ac), qout + 0.2);

    EXPECT_GT(actor.computeOutVF(sens, ac_empty), out - 0.25);
    EXPECT_LT(actor.computeOutVF(sens, ac_empty), out + 0.25);
    
    EXPECT_GT(actor2.computeOutVF(sens, ac_empty), out - 0.25);
    EXPECT_LT(actor2.computeOutVF(sens, ac_empty), out + 0.25);
    
    LOG_FILE("OptimizeNNTroughGradientOfAnotherNNFann.data", x1 << " " << actor.computeOutVF(sens, ac_empty) << " " <<
                                                              actor2.computeOutVF(sens, ac_empty)<< " " << out);
    //clear all; close all; X=load('OptimizeNNTroughGradientOfAnotherNNFann.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');plot(X(:,1),X(:,4), 'go');
    ac1_error += fabs(out - actor.computeOutVF(sens, ac_empty));
    ac2_error += fabs(out - actor2.computeOutVF(sens, ac_empty));
  }
  
  ac1_error /= 100;
  ac2_error /= 100;
  
  LOG_DEBUG(ac1_error << " " << ac2_error);
}

TEST(MLP, OptimizeNNTroughGradientOfAnotherNNFannMinDerivative) {
  MLP nn(2, {50}, 1, 0.0, true);
  MLP actor(1, {8}, 1);
  
  MLP actor2(actor);
  fann_set_activation_function_output(actor.getNeuralNet(), FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(actor2.getNeuralNet(), FANN_LINEAR);

  struct fann_train_data* data;
  if ( !boost::filesystem::exists( "OptimizeNNTroughGradientOfAnotherNNFannMinDerivative.cache.data" ) ){
    data = fann_create_train(200*200, 2, 1);
    
    
    for(uint n = 0; n < 200*200 ;n++){
      double x0 = bib::Utils::rand01()*2 -1;
      double x1 = bib::Utils::rand01()*2 -1;
      data->input[n][0]= x0;
      data->input[n][1]= x1;

      data->output[n][0]= x1*x1-(x0*x0)*x1;
    }
    
    
    nn.learn_stoch(data, 5800, 100, 0.0000001,1000);
    //nn.learn(data, 300, 0);
    
    fann_destroy_train(data);
    nn.save("OptimizeNNTroughGradientOfAnotherNNFannMinDerivative.cache.data");
  } else 
    nn.load("OptimizeNNTroughGradientOfAnotherNNFannMinDerivative.cache.data");
  

  
  datann_derivative d = {&nn, 1, 1};
  
  for(int i=0;i<1000; i++){
     data = fann_create_train(400, 1, 1);
//      uint n = 0;
//      auto iter2 = [&](const std::vector<double>& x) {
//         data->input[n][0]= x[0];
//         data->output[n][0]= x[0];//don't care
//        n++;
//      };
//      bib::Combinaison::continuous<>(iter2, 1, -1, 1, 200);
     for (uint n = 0; n < 400 ; n++){
       data->input[n][0]= bib::Utils::rand01()*2-1.;
       data->output[n][0]= 0;
     }
    
    
     fann_train_epoch_irpropm_gradient(actor.getNeuralNet(), data, derivative_nn, &d);

     fann_train_epoch_irpropm_gradient(actor2.getNeuralNet(), data, derivative_nn_inverting, &d);
     if(i%100 == 0)
      LOG_DEBUG(actor.weight_l1_norm() << " " << actor2.weight_l1_norm());
     
     fann_destroy_train(data);
  }
  
  // Test
  double ac1_error = 0;
  double ac2_error = 0;
  for (uint n = 0; n < 100 ; n++) {
    double x1 = bib::Utils::rand01()*2 -1;

    double out = -1.f; //not about derivative but constraints in [-1; 1]
    double qout = out*out-(x1*x1)*out;

    std::vector<double> sens(1);
    sens[0] = x1;
    std::vector<double> ac(1);
    std::vector<double> ac_empty(0);
    ac[0]= out;
    
    EXPECT_GT(nn.computeOutVF(sens, ac), qout - 0.25);
    EXPECT_LT(nn.computeOutVF(sens, ac), qout + 0.25);

    EXPECT_GT(actor.computeOutVF(sens, ac_empty), out - 0.2);
    EXPECT_LT(actor.computeOutVF(sens, ac_empty), out + 0.2);
    
    EXPECT_GT(actor2.computeOutVF(sens, ac_empty), out - 0.2);
    EXPECT_LT(actor2.computeOutVF(sens, ac_empty), out + 0.2);
    
    LOG_FILE("OptimizeNNTroughGradientOfAnotherNNFannMinDerivative.data", x1 << " " << actor.computeOutVF(sens, ac_empty) << " " <<
                                                              actor2.computeOutVF(sens, ac_empty)<< " " << out);
    //clear all; close all; X=load('OptimizeNNTroughGradientOfAnotherNNFannMinDerivative.data'); plot(X(:,1),X(:,2), '.'); hold on; plot(X(:,1),X(:,3), 'r.');plot(X(:,1),X(:,4), 'go');
    ac1_error += fabs(out - actor.computeOutVF(sens, ac_empty));
    ac2_error += fabs(out - actor2.computeOutVF(sens, ac_empty));
  }
  
  ac1_error /= 100;
  ac2_error /= 100;
  
  LOG_DEBUG(ac1_error << " " << ac2_error);
}
