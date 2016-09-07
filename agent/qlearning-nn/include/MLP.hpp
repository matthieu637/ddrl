#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <functional>
#include <thread>

#include "doublefann.h"
#include "fann_data.h"
#include <bib/Logger.hpp>
#include <bib/Converger.hpp>
#include <bib/Seed.hpp>
#include <bib/Utils.hpp>

typedef struct fann* NN;

#ifndef NO_OPTPP
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "opt++/newmat.h"
#include "opt++/NLF.h"
#include "opt++/BoundConstraint.h"
#include "opt++/LinearInequality.h"
#include "opt++/CompoundConstraint.h"
#include "opt++/OptNIPS.h"
#include "opt++/OptBCNewton.h"
#include "opt++/OptBaNewton.h"
#include "opt++/OptDHNIPS.h"

#include "UNLF2.hpp"
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

struct passdata {
  NN neural_net;
  const std::vector<double>& inputs;
};

void init_hs65_zero(int ndim, ColumnVector& x);
void init_hs65_random(int ndim, ColumnVector& x);
void hs65(int mode, int ndim, const ColumnVector& x, double& fx,
          ColumnVector& gx, SymmetricMatrix& Hx, int& result, void* data);


struct ParallelOptimization {
  ParallelOptimization(const NN _neural_net, const std::vector<double>& _inputs, const std::vector<double>& _init_search,
                       uint _size_motors, uint number_parra);
  ~ParallelOptimization();
  void operator()(const tbb::blocked_range<uint>& r) const;
  void free(uint expect_index);

 private:
  const NN neural_net;
  const std::vector<double>& inputs;
  const std::vector<double>& init_search;
  const uint ndim;
 public:
  std::vector < std::vector<double>* > a;
  std::vector<double>* cost;
};
#endif //NO_OPTPP

#define STANDARD_MLP
/*
 Lecun activation function is only used for hidden layer.
 For the last layer of approximation => useless
 For the last layer of motors => wrong sense (out of [-1 1] )
 */

class MLP {
 public:

 MLP(unsigned int input, const std::vector<uint>& hiddens, unsigned int sensors, double alpha, bool _lecun=false) : size_input_state(input),
    size_sensors(sensors), size_motors(size_input_state - sensors) {
#ifdef STANDARD_MLP
    std::vector<uint> layers;
    layers.push_back(input);
    for (auto i : hiddens)
      layers.push_back(i);
    layers.push_back(1);
    neural_net = fann_create_standard_array(layers.size(), layers.data());
    
    fann_set_training_algorithm(neural_net, FANN_TRAIN_INCREMENTAL);
    fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
#else
    neural_net = fann_create_shortcut(2, input, 1);
#endif

    fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC);

    fann_set_activation_function_output(neural_net, FANN_LINEAR);  // Linear cause Q(s,a) isn't normalized
    fann_set_learning_momentum(neural_net, 0.);
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
    
    fann_set_learning_rate(neural_net, alpha);
    fann_set_activation_steepness_hidden(neural_net, 0.5);
    fann_set_activation_steepness_output(neural_net, 1.);
    
    if(_lecun)
      lecun(input);
  }

  MLP(unsigned int input, unsigned int sensors) : size_input_state(input), size_sensors(sensors),
    size_motors(size_input_state - sensors) {

  }

  MLP(unsigned int input, const std::vector<uint>& hiddens, unsigned int motors, bool _lecun=false) : size_input_state(input), size_sensors(input),
    size_motors(motors) {
#ifdef STANDARD_MLP
    std::vector<uint> layers;
    layers.push_back(input);
    for (auto i : hiddens)
      layers.push_back(i);
    layers.push_back(motors);
    neural_net = fann_create_standard_array(layers.size(), layers.data());
#else
    neural_net = fann_create_shortcut(2, input, motors);
#endif
    
    fann_set_learning_momentum(neural_net, 0.);
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_LINEAR);
    fann_set_train_stop_function(neural_net, FANN_STOPFUNC_MSE);
    fann_set_activation_steepness_output(neural_net, 1.);
    fann_set_learning_rate(neural_net, 0.);
    fann_set_activation_function_output(neural_net, FANN_SIGMOID_SYMMETRIC);  // motor are normalized
    
    fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_steepness_hidden(neural_net, 0.5);

    if(_lecun){
      lecun(input);
    }
  }
  
  MLP(const MLP& m) : size_input_state(m.size_input_state), size_sensors(m.size_sensors), size_motors(m.size_motors){
      neural_net = fann_copy(m.neural_net);
  }

  void lecun(int input) {
//     fann_set_activation_steepness_hidden(neural_net, 2.f/3.f);
    fann_set_activation_steepness_hidden(neural_net, atanh(1.d/sqrt(3.d)));
    fann_set_activation_function_hidden(neural_net, FANN_SIGMOID_SYMMETRIC_LECUN);
 
    std::map<uint, uint> nb_inputs;
    uint neuron = 0;
    
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;

    for(layer_it = neural_net->first_layer; layer_it != neural_net->last_layer; layer_it++)
      for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){   
        nb_inputs[neuron] = neuron_it->last_con - neuron_it->first_con;
        neuron++;
      } 

    unsigned int source_index = 0;
    unsigned int destination_index = 0;

    for(layer_it = neural_net->first_layer; layer_it != neural_net->last_layer; layer_it++)
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            for (uint idx = neuron_it->first_con; idx < neuron_it->last_con; idx++){
                uint to_neuron = destination_index;
                double stddev = sqrt(nb_inputs[to_neuron])/2;
                if(nb_inputs[to_neuron] == 0)
                  stddev = sqrt(input)/2;
                
                std::normal_distribution<fann_type> dist(0, stddev);
                neural_net->weights[source_index] = (fann_type) dist(*bib::Seed::random_engine());

                source_index++;
            }    
            destination_index++;
        }
        
    //+normalization of inputs
    // mean on 0 + same covariance
  }

  virtual ~MLP() {
    fann_destroy(neural_net);
  }

  void learn(const std::vector<double>& sensors, const std::vector<double>& motors, double toval) {
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

  void learn(const std::vector<double>& inputs, const std::vector<double>& outputs) {
    fann_type* out = new fann_type[outputs.size()];
    for(uint j=0; j < outputs.size(); j++)
      out[j] = outputs[j];

    fann_type* in = new fann_type[inputs.size()];
    for (uint j = 0; j < inputs.size() ; j++)
      in[j] = inputs[j];

    fann_train(neural_net, in, out);
    delete[] in;
    delete[] out;
  }

  void learn(struct fann_train_data* data, uint max_epoch=10000, uint display_each = 0, double precision = 0.00001) {
    learn(neural_net, data, max_epoch, display_each, precision);
  }

  static void learn(NN neural_net, struct fann_train_data* data, uint max_epoch, uint display_each, double precision) {

    //FANN_TRAIN_BATCH FANN_TRAIN_RPROP
    fann_set_training_algorithm(neural_net, FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_TANH);

#ifdef STANDARD_MLP
    auto iter = [&]() {
      fann_train_epoch(neural_net, data);

    };
    auto eval = [&]() {
      return fann_get_MSE(neural_net);
    };

    //determinist is clearly non optimal
    bib::Converger::determinist<>(iter, eval, max_epoch, precision, display_each);
#else
    fann_cascadetrain_on_data(neural_net, data, 15, display_each, precision);
#endif
  }
  
  void learn_stoch(struct fann_train_data* data, uint max_epoch=10000, uint display_each = 0, double precision = 0.00001, uint stable_over=50){
    //FANN_TRAIN_BATCH FANN_TRAIN_RPROP
    fann_set_training_algorithm(neural_net, FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_TANH);

    auto iter = [&]() {
      fann_train_epoch(neural_net, data);
    };
    auto eval = [&]() {
      return fann_get_MSE(neural_net);
    };
    
    NN best_nn = nullptr;
    auto save_best = [&]() {
      if(best_nn != nullptr)
        fann_destroy(best_nn);
      best_nn = fann_copy(neural_net);
    };
    bib::Converger::min_stochastic<>(iter, eval, save_best, max_epoch, precision, display_each, stable_over);
    copy(best_nn);
    fann_destroy(best_nn); 
  }
  
  void learn_stoch_lw(struct fann_train_data* data, fann_type* lweight, uint max_epoch=10000, 
		      uint display_each = 0, double precision = 0.00001, uint stable_over=50){
    //FANN_TRAIN_BATCH FANN_TRAIN_RPROP
    fann_set_training_algorithm(neural_net, FANN_TRAIN_RPROP); //adaptive algorithm without learning rate
    fann_set_train_error_function(neural_net, FANN_ERRORFUNC_TANH);

    auto iter = [&]() {
      fann_train_epoch_lw(neural_net, data, lweight);
    };
    auto eval = [&]() {
      return fann_get_MSE(neural_net);
    };
    
    NN best_nn = nullptr;
    auto save_best = [&]() {
      if(best_nn != nullptr)
        fann_destroy(best_nn);
      best_nn = fann_copy(neural_net);
    };
    bib::Converger::min_stochastic<>(iter, eval, save_best, max_epoch, precision, display_each, stable_over);
    copy(best_nn);
    fann_destroy(best_nn); 
  }

  double computeOutVF(const std::vector<double>& sensors, const std::vector<double>& motors) const {
    return computeOutVF(neural_net, sensors, motors);
  }

  static double computeOutVF(NN neural_net, const std::vector<double>& sensors, const std::vector<double>& motors) {
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

  std::vector<double>* computeOut(const std::vector<double>& in) const {
    return computeOut(neural_net, in);
  }

  static std::vector<double>* computeOut(NN neural_net, const std::vector<double>& in) {
    uint m = in.size();

    fann_type* inputs = new fann_type[m];
    for (uint j = 0; j < m ; j++)
      inputs[j] = in[j];

    fann_type* out = fann_run(neural_net, inputs);

    std::vector<double>* outputs = new std::vector<double>(fann_get_num_output(neural_net));
    for(uint j=0; j < outputs->size(); j++)
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
   * @return std::vector< double >* your job to delete it
   */

#ifdef NO_OPTPP
  virtual std::vector<double>* optimized(const std::vector<double>&, const std::vector<double>&, uint) const {
	LOG_ERROR("not implemented : OPTPP optional");
	exit(1);
	return nullptr;
  }
#else
  virtual std::vector<double>* optimized(const std::vector<double>& inputs, const std::vector<double>& init_search = {},
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
#endif
  
  std::vector<double>* optimizedBruteForce(const std::vector<double>& inputs, double discre=0.1){
    std::vector<double> motors(1);

    motors[0] = -1.f;
    double imax = -1.f;
    double vmax = computeOutVF(inputs, motors);
    
    for(double a=-1+discre; a <= 1; a+=discre){
      motors[0] = a;
      double val = computeOutVF(inputs, motors);
      if(val > vmax){
        vmax = val;
        imax = a;
      }
    }

    std::vector<double>* outputs = new std::vector<double>(size_motors);
//     for(uint j=0; j < outputs->size(); j++)
//       outputs->at(j) = out[j];
    outputs->at(0) = imax;
    return outputs;
    
  }
  
  void print(){
      struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(neural_net), sizeof(struct fann_connection));

      fann_get_connection_array(neural_net, connections);

      for(uint j=0; j<fann_get_total_connections(neural_net); j++)
          std::cout << connections[j].weight << " ";
      std::cout << std::endl;

      free(connections);
  }
  
  uint getNumberHiddenNeurons(){
    std::vector<uint> layers(fann_get_num_layers(neural_net));
    
    //struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_num_layers(neural_net), sizeof(unsigned int));
    
    fann_get_layer_array(neural_net, layers.data());

    uint sum = 0;
    for(uint i=1;i<layers.size()-1;i++){
      sum += layers[i]-1;//minus bias
    }
    
    return sum;
  }
  
  void getHiddenNeurons(std::vector<double>& neurons){
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    struct fann_neuron *neuron_it_end;
    struct fann_layer *layer_it_end = neural_net->last_layer;
    
    /* The following assumes that the last unused bias has no connections */
    uint i = 0;
    /* for each layer */
    layer_it = neural_net->first_layer;
    layer_it++;
    --layer_it_end;
    for(; layer_it != layer_it_end; layer_it++){
        /* for each neuron */
        neuron_it_end = layer_it->last_neuron;
        --neuron_it_end;
        
        for(neuron_it = layer_it->first_neuron; neuron_it != neuron_it_end; neuron_it++){
            neurons[i++] = neuron_it->value;
        }
    }
  }
  
  void initializeFrom(MLP* _old){
    NN old = _old->neural_net;
    
    struct fann_connection* new_connections = (struct fann_connection*) calloc(fann_get_total_connections(neural_net), sizeof(struct fann_connection));
    struct fann_connection* old_connections = (struct fann_connection*) calloc(fann_get_total_connections(old), sizeof(struct fann_connection));
    
    fann_get_connection_array(neural_net, new_connections);
    fann_get_connection_array(old, old_connections);
    
    std::vector<uint> new_layers(fann_get_num_layers(neural_net));
    std::vector<uint> old_layers(fann_get_num_layers(old));
    
    fann_get_layer_array(neural_net, new_layers.data());
    fann_get_layer_array(old, old_layers.data());
    
    uint new_index=0;
    uint old_index=0;
    ASSERT(new_layers.size() == old_layers.size(), "same layer size for now");
    for(uint l=0; l < new_layers.size() - 1; l++){
      for(uint n=0 ; n < new_layers[l] ; n++) {
        for(uint c=0; c < new_layers[l+1] ; c++){
            if(n < old_layers[l] && c < old_layers[l+1]){
              new_connections[new_index].weight = old_connections[old_index].weight;
              old_index++;
            }
            new_index++;
            
        }
      }
    }
    
    fann_set_weight_array(neural_net, new_connections, fann_get_total_connections(neural_net));
    
    delete new_connections;
    delete old_connections;
  }
  
  static unsigned long long GetHashForDouble(double x, int discretization)
  {
    x = bib::Utils::transform(x, -2,2,0, discretization);
    unsigned long long y = x;
    return y;
  }
  
  double hash(){
      struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(neural_net), sizeof(struct fann_connection));

      for(uint j=0; j<fann_get_total_connections(neural_net); j++)
          connections[j].weight=0;

      fann_get_connection_array(neural_net, connections);

      int discretization = exp(log(std::numeric_limits<unsigned long long>::max())/(fann_get_total_connections(neural_net)+1));
      
//       LOG_DEBUG(discretization);
      unsigned long long ret = 0;
      unsigned long long max = 0;
      for(uint j=0; j<fann_get_total_connections(neural_net); j++){
        ret += pow(discretization, j)*(GetHashForDouble(connections[j].weight, discretization));
        max += pow(discretization, j)*discretization;
      }

      free(connections);
      
      double f = ret;
      f = f / ((double)max);
      return f;
  }
  
  uint getNbParameters(){
      return fann_get_total_connections(neural_net);
  }

  void copy(NN new_nn) {
    fann_destroy(neural_net);
    neural_net = fann_copy(new_nn);
  }

  void save(const std::string& path) const {
    fann_save(neural_net, path.c_str());
  }

  void load(const std::string& path) {
    fann_destroy(neural_net);
    neural_net = fann_create_from_file(path.c_str());
  }

  NN getNeuralNet() {
    return neural_net;
  }

  double error() {
    return fann_get_MSE(neural_net);
  }
  
  double weight_l1_norm(){
    return fann_get_l1_norm(neural_net);
  }
  
  double weightSum(){
#ifndef NDEBUG
    return fann_get_l1_norm(neural_net);
#else
    return 0;
#endif
  }

 protected:
  NN neural_net;
  unsigned int size_input_state;
  unsigned int size_sensors;
  unsigned int size_motors;
};


struct datann_derivative {
  MLP* nn;
  int n; //number of s
  int m; //number of a
};

double derivative_nn_easy(double*, double *, int, void*);

double derivative_nn(double* input, double *neuron_value, int a_dim, void* data);
double derivative_nn_inverting(double* input, double *neuron_value, int a_dim, void* data);

#endif  // MLP_H

