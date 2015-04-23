#ifndef MLP_H
#define MLP_H

#include <vector>
#include <functional>
#include <thread>

#include "doublefann.h"
#include "opt++/newmat.h"
#include "opt++/NLF.h"
#include "opt++/BoundConstraint.h"
#include "opt++/LinearInequality.h"
#include "opt++/CompoundConstraint.h"
#include "opt++/OptNIPS.h"
#include "opt++/OptBCNewton.h"
#include "opt++/OptBaNewton.h"
#include "opt++/OptDHNIPS.h"
#include <fann_data.h>

//     OptNewton(&nlp): Newton method for unconstrained problems
//     OptBCNewton(&nlp): Newton method for bound-constrained problems
//     OptBaNewton(&nlp): Newton method for bound-constrained problems
//     OptNIPS(&nlp): nonlinear interior-point method for generally constrained problems
#include "UNLF2.hpp"
#include "bib/Logger.hpp"

using OPTPP::Constraint;
using OPTPP::CompoundConstraint;
using OPTPP::NLF2;
using OPTPP::UNLF2;
using OPTPP::OptBaNewton;
using OPTPP::BoundConstraint;
using OPTPP::NLPFunction;
using OPTPP::NLPGradient;
using OPTPP::NLPHessian;
using OPTPP::OptNIPS;
using OPTPP::OptBCNewton;

using NEWMAT::ColumnVector;
using NEWMAT::Matrix;
using NEWMAT::SymmetricMatrix;

typedef struct fann* NN;
struct passdata {
    NN neural_net;
    const std::vector<float>& inputs;
};

void init_hs65_zero(int ndim, ColumnVector& x);
void init_hs65_random(int ndim, ColumnVector& x);
void hs65(int mode, int ndim, const ColumnVector& x, double& fx,
          ColumnVector& gx, SymmetricMatrix& Hx, int& result, void* data);


struct ParallelOptimization {
    ParallelOptimization(const NN _neural_net, const std::vector<float>& _inputs, const std::vector<float>& _init_search,
                         uint _size_motors, uint number_parra);
    ~ParallelOptimization();
//     void operator()(const tbb::blocked_range<uint>& r) const;
    void operator()(uint);

private:
    const NN neural_net;
    const std::vector<float>& inputs;
    const std::vector<float>& init_search;
    const uint ndim;
public:
    std::vector < std::vector<float>* > a;
    std::vector<double>* cost;
};

class MLP
{
public:

    MLP(unsigned int input, unsigned int hidden, unsigned int sensors, float alpha) : size_input_state(input),
        size_sensors(sensors), size_motors(size_input_state - sensors) {
        neural_net = fann_create_standard(3, input, hidden, 1);

        fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC);

        fann_set_activation_function_output(neural_net, FANN_LINEAR);  // Linear cause Q(s,a) isn't normalized
        fann_set_learning_momentum(neural_net, 0.);
        fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
        fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
        fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
        fann_set_learning_rate(neural_net, alpha);
        fann_set_activation_steepness_hidden(neural_net, 0.5);
        fann_set_activation_steepness_output(neural_net, 1.);
    }

    ~MLP() {
        fann_destroy(neural_net);
    }

    void learn(const std::vector<float>& sensors, const std::vector<float>& motors, float toval) {
        fann_type out[1];
        out[0] = toval;

        uint m = sensors.size();
        uint n = motors.size();
        fann_type* inputs = new fann_type[m + n];
        for (uint j = 0; j < m ; j++)
            inputs[j] = sensors[j];
        for (uint j = m; j < m + n; j++)
            inputs[j] = motors[j - m];

        fann_train(neural_net, inputs, out);
        delete[] inputs;
    }

    double computeOut(const std::vector<float>& sensors, const std::vector<float>& motors) {
        uint m = sensors.size();
        uint n = motors.size();
        fann_type* inputs = new fann_type[m + n];
        for (uint j = 0; j < m ; j++)
            inputs[j] = sensors[j];
        for (uint j = m; j < m + n; j++)
            inputs[j] = motors[j - m];

        fann_type* out = fann_run(neural_net, inputs);
        delete[] inputs;
        return out[0];
    }

    std::vector<float>* optimized(const std::vector<float>& inputs, const std::vector<float>& init_search = {}) {
        uint number_parra_eval = 8;

        ParallelOptimization para(neural_net, inputs, init_search, size_motors, number_parra_eval);

        vector<std::thread*> threads(number_parra_eval);
        for (uint thread = 0; thread  < number_parra_eval; thread++) {
            //careful to give a point of para or it'll be deleted
            threads[thread] = new std::thread(&ParallelOptimization::operator(), &para, thread);
        }

        for (uint thread = 0; thread  < number_parra_eval; thread++) {
            threads[thread]->join();
            delete threads[thread];
        }

        double imin = 0;
        for (uint i = 1; i < number_parra_eval; i++)
            if (para.cost->at(imin) > para.cost->at(i))
                imin = i;

        std::vector<float>* best_ac = new std::vector<float>(*para.a[imin]);

//         for(uint i=0;i < number_parra_eval; i++)
//           LOG_DEBUG("all solutions : " << para.a[i]->at(0) << " " << para.cost->at(i));

        return best_ac;
    }

    void save(const std::string& path) const {
        fann_save(neural_net, path.c_str());
    }

    void load(const std::string& path) {
        neural_net = fann_create_from_file(path.c_str());
    }

    NN getNeuralNet() {
        return neural_net;
    }

private:
    NN neural_net;
    unsigned int size_input_state;
    unsigned int size_sensors;
    unsigned int size_motors;
};

#endif  // MLP_H
