#include "MLP.hpp"

#include <vector>
#include "doublefann.h"
#include "bib/Assert.hpp"

#define activation_function(x, lambda) tanh(lambda * x)

void init_hs65(int ndim, ColumnVector& x) {
  for (uint i = 1; i <= (uint) ndim; i++)
    x(i) = 0;
}

class my_weights {
  typedef struct fann_connection sfn;

 public:
  my_weights(NN neural_net, const std::vector<float>& sensors, const ColumnVector& x, uint _m, uint _n) : m(_m), n(_n) {
    _lambda = fann_get_activation_steepness(neural_net, 1, 0);

    unsigned int number_connection = fann_get_total_connections(neural_net);
    connections = reinterpret_cast<sfn*>(calloc(number_connection, sizeof(sfn)));

    fann_get_connection_array(neural_net, connections);

    uint number_layer = fann_get_num_layers(neural_net);
    layers = reinterpret_cast<uint*>(calloc(number_layer, sizeof(sfn)));

    fann_get_layer_array(neural_net, layers);
    ASSERT(number_layer == 3, number_layer);

    for (uint i = 0; i < h() * (m+n+1); i++) {
      _ASSERT_EQ(connections[i].from_neuron, i % (m+n+1));
      _ASSERT_EQ(connections[i].to_neuron, (m+n+1) + i / (m+n+1));
    }

    Ci.clear();
    Ci.resize(h());
    for (uint i = 0; i < h(); i++) {
      Ci[i] = 0;
      for (uint j = 0; j < m; j++)
        Ci[i] += sensors[j] * w(j, i);

      Ci[i] += connections[ i * (m+n+1) + m + n].weight;
      _ASSERT_EQ(connections[ i * (m+n+1) + m + n].from_neuron, m+n);
      _ASSERT_EQ(connections[ i * (m+n+1) + m + n].to_neuron, m+n+1+i);
    }

    Di.clear();
    Di.resize(h());
    for (uint i = 0; i < h(); i++) {
      Di[i] = Ci[i];
      for (uint j = 1; j <= n; j++)
        Di[i] += x(j) * w(m+j-1, i);
    }

    ASSERT(number_connection == h()*(m+n+1) + (h() + 1), "");
  }

  ~my_weights() {
    free(connections);
    free(layers);
  }

  double v(uint i) const {
    sfn conn = connections[ h() * (m+n+1) + i];

    _ASSERT_EQS(conn.from_neuron, (m+n+1) + i, " i: "<< i  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m+n+1) + (h() + 1), " i: "<< i  << " m " << m << " n " << n << " h " << h());
    return conn.weight;
  }

  double w(uint j, uint i) const {
    sfn conn = connections[ i * (m+n+1) + j];

    _ASSERT_EQS(conn.from_neuron, j, " i: "<< i << " j : "<< j  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m+n+1) + i, " i: "<< i << " j : "<< j  << " m " << m << " n " << n << " h " << h());
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

void hs65(int mode, int ndim, const ColumnVector& x, double& fx,
          ColumnVector& gx, SymmetricMatrix& Hx, int& result, void* data) {
  passdata* d = static_cast<passdata*>(data);
  NN neural_net = d->neural_net;

  uint n = ndim;
  uint m = fann_get_num_input(neural_net) - n;

  if (mode & NLPFunction) {
    fann_type* inputs = new fann_type[m + n];
    for (uint j = 0; j < m ; j++)
      inputs[j] = d->inputs[j];
    for (uint j = m; j < m + n; j++)
      inputs[j] = x(j - m + 1);

    fann_type* out = fann_run(neural_net, inputs);
    // negative to maximaze instead of minimizing
    fx = - out[0];
    result = NLPFunction;
    delete[] inputs;
  }

  if (mode & NLPGradient) {
    my_weights _w(neural_net, d->inputs, x, m, n);
    for (uint j = 1; j <= n; j++) {
      gx(j) = 0;
      for (uint i = 0; i < _w.h() ; i++) {
        double der = activation_function(_w.D(i), _w.lambda());
        gx(j) = gx(j) - _w.v(i) * _w.w(m + j - 1, i) * _w.lambda() * (1.0 - der * der);
      }
    }
    result = NLPGradient;
  }

  if (mode & NLPHessian) {
    my_weights _w(neural_net, d->inputs, x, m, n);
    for (uint j = 1; j <= n; j++) {
      for (uint k = 1; k <= n; k++) {
        Hx(j,k) = 0;
        for (uint i = 0; i < _w.h() ; i++) {
          double der = activation_function(_w.D(i), _w.lambda());
          double Lambda_ij = _w.v(i) * _w.w(m + j - 1, i) * _w.lambda();
          Hx(j, k) = Hx(j, k) - -2. * _w.lambda() * _w.w(m + k - 1, i) * Lambda_ij * der * (1.0 - der * der);
        }
      }
    }
    result = NLPHessian;
  }
}
