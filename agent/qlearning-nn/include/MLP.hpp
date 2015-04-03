#ifndef MLP_H
#define MLP_H

#include "doublefann.h"

class MLP
{
public:
    typedef struct fann* NN;

    MLP(unsigned int input, unsigned int hidden, unsigned int output) {
        neural_net = fann_create_standard(3, input, hidden, output);

        fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC);

        fann_set_activation_function_output(neural_net, FANN_LINEAR);  // Linear cause Q(s,a) isn't normalized
        fann_set_learning_momentum(neural_net, 0.);
        fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
        fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
        fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
//         fann_set_learning_rate(neural_net, this->param.alpha);
    }

private:
    NN neural_net;
};

#endif // MLP_H
