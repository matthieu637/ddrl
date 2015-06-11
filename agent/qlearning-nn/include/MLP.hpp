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
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "UNLF2.hpp"
#include "bib/Logger.hpp"
#include <bib/Converger.hpp>

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
  void operator()(const tbb::blocked_range<uint>& r) const;
  void free(uint expect_index);

 private:
  const NN neural_net;
  const std::vector<float>& inputs;
  const std::vector<float>& init_search;
  const uint ndim;
 public:
  std::vector < std::vector<float>* > a;
  std::vector<double>* cost;
};

class MLP {
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
  
  MLP(unsigned int input, unsigned int sensors) : size_input_state(input), size_sensors(sensors), size_motors(size_input_state - sensors){
     
  }

  virtual ~MLP() {
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
  
  void learn(const std::vector<float>& inputs, const std::vector<float>& outputs) {
    fann_type* out = new fann_type[outputs.size()];
    for(uint j=0;j < outputs.size();j++)
      out[j] = outputs[j];

    fann_type* in = new fann_type[inputs.size()];
    for (uint j = 0; j < inputs.size() ; j++)
      in[j] = inputs[j];

    fann_train(neural_net, in, out);
    delete[] in;
    delete[] out;
  }

  void learn(struct fann_train_data* data, uint max_epoch=10000, uint display_each = 0, float precision = 0.00001) {
    learn(neural_net, data, max_epoch, display_each, precision);
  }

  static void learn(NN neural_net, struct fann_train_data* data, uint max_epoch, uint display_each, float precision) {
    fann_set_training_algorithm(neural_net, FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_TANH);

    auto iter = [&]() {
      fann_train_epoch(neural_net, data);
    };
    auto eval = [&]() {
      return fann_get_MSE(neural_net);
    };

    bib::Converger::determinist<>(iter, eval, max_epoch, precision, display_each);
  }

  double computeOut(const std::vector<float>& sensors, const std::vector<float>& motors) const {
    return computeOut(neural_net, sensors, motors);
  }

  static double computeOut(NN neural_net, const std::vector<float>& sensors, const std::vector<float>& motors) {
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
  
  std::vector<float>* computeOut(const std::vector<float>& in) const {
    return computeOut(neural_net, in);
  }
  
  static std::vector<float>* computeOut(NN neural_net, const std::vector<float>& in) {
    uint m = in.size();

    fann_type* inputs = new fann_type[m];
    for (uint j = 0; j < m ; j++)
      inputs[j] = in[j];

    fann_type* out = fann_run(neural_net, inputs);
    std::vector<float>* outputs = new std::vector<float>(fann_get_num_output(neural_net));
    for(uint j=0;j < outputs->size();j++)
      outputs->at(j) = out[j];
    delete[] inputs;
    return outputs;
  }

  /**
   * @brief Search the input in [-1; 1 ] that maximize the network output
   * Can be traped in local maxima
   *
   * @param inputs part of the input layer known
   * @param init_search initial solution for the input layer
   * @param nb_solution kept it empty if you do not care to be traped in local maxima
   * If you want to try to found a global maxima you can increase this number however there is no garanty
   *
   * @return std::vector< float >* your job to delete it
   */

  virtual std::vector<float>* optimized(const std::vector<float>& inputs, const std::vector<float>& init_search = {},
                                uint nb_solution = 12) const {

    ParallelOptimization para(neural_net, inputs, init_search, size_motors, nb_solution);
    tbb::parallel_for(tbb::blocked_range<uint>(0, nb_solution), para);

    double imin = 0;
    for (uint i = 1; i < nb_solution; i++)
      if (para.cost->at(imin) > para.cost->at(i))
        imin = i;

//         for(uint i=0;i < nb_solution; i++)
//           LOG_DEBUG("all solutions : " << para.a[i]->at(0) << " " << para.cost->at(i) << " " << i);

//         bib::Logger::PRINT_ELEMENTS<>(*best_ac);
    para.free(imin);

    return para.a[imin];
  }

  void copy(NN new_nn) {
    fann_destroy(neural_net);
    neural_net = fann_copy(new_nn);
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

  double error() {
    return fann_get_MSE(neural_net);
  }

 protected:
  NN neural_net;
  unsigned int size_input_state;
  unsigned int size_sensors;
  unsigned int size_motors;
};

#endif  // MLP_H
