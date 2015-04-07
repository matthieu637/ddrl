#ifndef MLP_H
#define MLP_H

//OPT++ param
#define HAVE_NAMESPACES
#define HAVE_STD

#include <vector>
#include <functional>

#include "doublefann.h"
#include "opt++/NLF.h"
#include "opt++/BoundConstraint.h"
#include "opt++/LinearInequality.h"
#include "opt++/CompoundConstraint.h"
#include "opt++/OptNIPS.h"
//     OptNewton(&nlp): Newton method for unconstrained problems
//     OptBCNewton(&nlp): Newton method for bound-constrained problems
//     OptBaNewton(&nlp): Newton method for bound-constrained problems
//     OptNIPS(&nlp): nonlinear interior-point method for generally constrained problems

using namespace OPTPP;

using NEWMAT::ColumnVector;
using NEWMAT::Matrix;
using NEWMAT::SymmetricMatrix;

typedef struct fann* NN;
struct passdata{
    NN neural_net;
    const std::vector<float>& inputs;
};

void init_hs65(int ndim, ColumnVector& x);
void hs65(int mode, int ndim, const ColumnVector& x, double& fx,
          ColumnVector& gx, SymmetricMatrix& Hx, int& result, void* data);

class MLP
{
public:

    MLP(unsigned int input, unsigned int hidden, unsigned int sensors, float alpha) : size_input_state(input), size_sensors(sensors), size_motors(size_input_state - sensors) {
        neural_net = fann_create_standard(3, input, hidden, 1);

        fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC);

        fann_set_activation_function_output(neural_net, FANN_LINEAR);  // Linear cause Q(s,a) isn't normalized
        fann_set_learning_momentum(neural_net, 0.);
        fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
        fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
        fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
        fann_set_learning_rate(neural_net, alpha);
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
        delete inputs;
    }
    
    double computeOut(const std::vector<float>& sensors, const std::vector<float>& motors){
        uint m = sensors.size();
        uint n = motors.size();
        fann_type* inputs = new fann_type[m + n];
        for (uint j = 0; j < m ; j++)
            inputs[j] = sensors[j];
        for (uint j = m; j < m + n; j++)
            inputs[j] = motors[j - m];

        fann_type* out = fann_run(neural_net, inputs);
        delete inputs;
        return out[1];
    }

    std::vector<float>* optimized(const std::vector<float>& inputs) {
        std::vector<float>* a = new std::vector<float>(size_motors);

        uint ndim = size_motors;
        ColumnVector lower(ndim), upper(ndim);

        for (uint i = 0; i < ndim; i++) {
            lower << -1.0f ;
            upper <<  1.0f ;
        }

        Constraint c1 = new BoundConstraint(ndim, lower, upper);
        CompoundConstraint* constraints = new CompoundConstraint(c1);

        passdata d = {neural_net, inputs};
        NLF2 nips(ndim, hs65, init_hs65, constraints, &d);

        OptNIPS objfcn(&nips);

// The "0" in the second argument says to create a new file.  A "1"
// would signify appending to an existing file.

        objfcn.setOutputFile("/dev/null", 1);
        objfcn.setFcnTol(1.0e-06);
        objfcn.setMaxIter(20);
        objfcn.setMeritFcn(ArgaezTapia);

        objfcn.optimize();

        //objfcn.printStatus("Solution from nips");
        //objfcn.cleanup();
        
        for(uint i=0;i<a->size();i++)
          a->at(i) = objfcn.getXPrev()(i+1);

        return a;
    }


private:
    NN neural_net;
    unsigned int size_input_state;
    unsigned int size_sensors;
    unsigned int size_motors;
};

#endif // MLP_H
