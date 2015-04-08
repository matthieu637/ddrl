#include "gtest/gtest.h"

#include <vector>
#include "MLP.hpp"
#include <bib/Utils.hpp>
#include <iostream>

float OR(float x1, float x2)
{
    if (x1 == 1.f || x2 == 1.f)
        return 1.f;
    return -1.f;
}

float AND(float x1, float x2)
{
    if (x1 == 1.f && x2 == 1.f)
        return 1.f;
    return -1.f;
}

TEST(MLP, ConsistentActivationFunction)
{
    MLP nn(1, 1, 0, 0.01f);

    fann_set_weight(nn.getNeuralNet(), 0, 2, 1.f);
    fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
    fann_set_weight(nn.getNeuralNet(), 2, 4, 1.f);
    fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

    std::vector<float> sens(0);
    std::vector<float> ac(1);
    ac[0] = 1;

    double lambda = 0.5;
    fann_set_activation_steepness_output(nn.getNeuralNet(), lambda);
    fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), tanh(lambda * 1.) / 2.);

    lambda = 0.8;
    fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), tanh(lambda * 1.) / 2.);

    ac[0] = -1;
    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), tanh(lambda * -1.) / 2.);

    lambda = 0.5;
    fann_set_activation_steepness(nn.getNeuralNet(), lambda, 1, 0);
    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), tanh(lambda * -1.) / 2.);

    fann_set_weight(nn.getNeuralNet(), 0, 2, 0.f);
    fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);
    fann_set_weight(nn.getNeuralNet(), 2, 4, 0.f);
    fann_set_weight(nn.getNeuralNet(), 3, 4, 5.f);

    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), 5.f / 2.f);

    fann_set_weight(nn.getNeuralNet(), 0, 2, 0.2f);
    fann_set_weight(nn.getNeuralNet(), 1, 2, 0.4f);
    fann_set_weight(nn.getNeuralNet(), 2, 4, 0.6f);
    fann_set_weight(nn.getNeuralNet(), 3, 4, 0.8f);

    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), (tanh(lambda * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f) / 2.f);

    fann_set_activation_steepness_output(nn.getNeuralNet(), 1.f);
    EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac),  tanh(lambda * (-1. * 0.2f + 0.4f)) * 0.6f + 0.8f);
}


TEST(MLP, LearnOpposite)
{
    MLP nn(1, 1, 0, 0.5f);

    fann_set_weight(nn.getNeuralNet(), 0, 2, -1.f);
    fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);

    fann_set_weight(nn.getNeuralNet(), 2, 4, 1. / tanh(0.5 * 1.));
    fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

    //Test
    for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = x1 == 1.f ? -1.f : 1.f;

        std::vector<float> sens(0);
        std::vector<float> ac(1);
        ac[0] = x1;

        EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), out);
    }

    // Learn
    for (uint n = 0; n < 1000 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = x1 == 1.f ? -1.f : 1.f;

        std::vector<float> sens(0);
        std::vector<float> ac(1);
        ac[0] = x1;

        nn.learn(sens, ac, out);
    }

    //Test
    for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = x1 == 1.f ? -1.f : 1.f;

        std::vector<float> sens(0);
        std::vector<float> ac(1);
        ac[0] = x1;

        EXPECT_EQ(nn.computeOut(sens, ac), out);
    }
}


TEST(MLP, LearnAndOr)
{
    MLP nn(3, 10, 2, 0.5f);

    // Learn
    for (uint n = 0; n < 10000 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;
        double x2 = bib::Utils::randBool() ? 1.f : -1.f;
        double x3 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = AND(OR(x1, x2), x3);

        std::vector<float> sens(2);
        sens[0] = x1;
        sens[1] = x2;
        std::vector<float> ac(1);
        ac[0] = x3;

        nn.learn(sens, ac, out);
    }

    //Test
    for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;
        double x2 = bib::Utils::randBool() ? 1.f : -1.f;
        double x3 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = AND(OR(x1, x2), x3);

        std::vector<float> sens(2);
        sens[0] = x1;
        sens[1] = x2;
        std::vector<float> ac(1);
        ac[0] = x3;

        EXPECT_GT(nn.computeOut(sens, ac), out - 0.02);
        EXPECT_LT(nn.computeOut(sens, ac), out + 0.02);
    }
}

TEST(MLP, OptimizeOpposite)
{
    MLP nn(1, 1, 0, 0.5f);

    fann_set_weight(nn.getNeuralNet(), 0, 2, -1.f);
    fann_set_weight(nn.getNeuralNet(), 1, 2, 0.f);

    fann_set_weight(nn.getNeuralNet(), 2, 4, 1. / tanh(0.5 * 1.));
    fann_set_weight(nn.getNeuralNet(), 3, 4, 0.f);

    //Test
    std::vector<float> sens(0);
    for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = x1 == 1.f ? -1.f : 1.f;

        std::vector<float> ac(1);
        ac[0] = x1;

        EXPECT_DOUBLE_EQ(nn.computeOut(sens, ac), out);
    }

    std::vector<float>* ac = nn.optimized(sens);
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

        std::vector<float> sens(0);
        std::vector<float> ac(1);
        ac[0] = x1;

        nn.learn(sens, ac, out);
    }

    ac = nn.optimized(sens);
    EXPECT_GT(ac->at(0), -1. - 0.01);
    EXPECT_LT(ac->at(0), -1. + 0.01);
    delete ac;
}

TEST(MLP, MLPCheckOuput)
{
    MLP nn(1, 1, 2, 0.5f);
    std::vector<float> sens(0);

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

TEST(MLP, MLPCheckDerivative)
{
    MLP nn(1, 1, 2, 0.5f);
    std::vector<float> sens(0);

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

TEST(MLP, MLPCheckDerivativeHard)
{
    MLP nn(4, 10, 3, 0.5f);
    std::vector<float> sens(3);
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

TEST(MLP, MLPCheckHessian)
{
    MLP nn(4, 10, 3, 0.5f);
    std::vector<float> sens(3);
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

TEST(MLP, OptimizeAndOr)
{
    MLP nn(3, 10, 2, 0.5f);

    // Learn
    for (uint n = 0; n < 10000 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;
        double x2 = bib::Utils::randBool() ? 1.f : -1.f;
        double x3 = bib::Utils::randBool() ? 1.f : -1.f;

        double out = AND(OR(x1, x2), x3);

        std::vector<float> sens(2);
        sens[0] = x1;
        sens[1] = x2;
        std::vector<float> ac(1);
        ac[0] = x3;

        nn.learn(sens, ac, out);
    }



    //Test
    for (uint n = 0; n < 100 ; n++) {
        double x1 = bib::Utils::randBool() ? 1.f : -1.f;
        double x2 = bib::Utils::randBool() ? 1.f : -1.f;

        std::vector<float> sens(2);
        sens[0] = x1;
        sens[1] = x2;

        std::vector<float>* ac = nn.optimized(sens);
        double his_sol = nn.computeOut(sens, *ac);

        ac->at(0) = 1.f;
        double my_sol = nn.computeOut(sens, *ac);

        EXPECT_GE(his_sol, my_sol - 0.01);
        delete ac;
    }
}
