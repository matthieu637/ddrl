#ifndef LINMLP_H
#define LINMLP_H

#include "MLP.hpp"

class LinMLP : public MLP
{
public:
    LinMLP(uint input, uint output, float alpha): MLP(input, input) {
        neural_net = fann_create_standard(2, input, output);

        fann_set_activation_function_output(neural_net, FANN_SIGMOID_SYMMETRIC);
        fann_set_learning_momentum(neural_net, 0.);
        fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
        fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
        fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
        fann_set_activation_steepness_output(neural_net, 1.);
        fann_set_learning_rate(neural_net, alpha);
        fann_randomize_weights(neural_net, -0.3, 0.3);
    }
    
    

};

#endif
