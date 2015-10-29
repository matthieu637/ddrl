#ifndef LINMLP_H
#define LINMLP_H

#include "MLP.hpp"

class LinMLP : public MLP {
 public:
  LinMLP(uint input, uint output, float alpha, bool _lecun=false): MLP(input, input) {
    neural_net = fann_create_standard(2, input, output);

    fann_set_learning_momentum(neural_net, 0.);
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
    fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
    fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
    fann_set_activation_steepness_output(neural_net, 1.);
    fann_set_learning_rate(neural_net, alpha);
    fann_set_activation_function_output(neural_net, FANN_SIGMOID_SYMMETRIC);

    if(_lecun){
      lecun(input);
      fann_set_activation_steepness_output(neural_net, atanh(1.d/sqrt(3.d)));
      fann_set_activation_function_output(neural_net, FANN_SIGMOID_SYMMETRIC_LECUN);
      _lecun_check = true;
    }
  }



};

#endif
