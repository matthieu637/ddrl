#include "MLP.hpp"

#include "doublefann.h"
#include <fann_data.h>
#include "bib/Assert.hpp"


#define activation_function(x, lambda) (2.0f/(1.0f + exp(- lambda * x)) - 1.0f)
//#define activation_function(x, lambda) (2.0f/(1.0f + exp(- 2 * lambda * x)) - 1.0f)

void init_hs65(int ndim, ColumnVector& x)
{
    for (uint i = 1; i <= (uint) ndim; i++)
        x(i) = 0;
}

class my_weights
{
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

        Ci.clear();
        Ci.resize(h());
        for (uint i = 0; i < h(); i++) {
            Ci[i] = 0;
            for (uint j = 0; j < m; j++)
                Ci[i] += sensors[j] * connections[i + h() * j].weight;

            Ci[i] += connections[ i + h() * (m + n)].weight;
        }

        Di.clear();
        Di.resize(h());
        for (uint i = 0; i < h(); i++) {
            Di[i] = Ci[i];
            for (uint j = 1; j <= n; j++)
                Di[i] += x(j) * connections[i + h() * (j - 1 + m)].weight;
        }
        
        ASSERT(number_connection == h()*(m+n+1) + (h() + 1), "h : " << h() << " , m + n + 1 : " << m+n+1 << " , nb con : " << number_connection << " , n :" << n );
    }

    ~my_weights() {
        free(connections);
        free(layers);
    }

    double v(uint i) {
        return connections[h() * (m + n + 1) + i].weight;
    }

    double w(uint j, uint i) {
        return connections[i + (h() * j)].weight;
    }

    uint h() {
        return layers[1];
    }

    double C(uint i) {
        return Ci[i];
    }

    double D(uint i) {
        return Di[i];
    }

    double lambda() {
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
          ColumnVector& gx, SymmetricMatrix& Hx, int& result, void* data)
{
    passdata* d = (passdata*) data;
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
        fx = out[1];
        result = NLPFunction;
        delete[] inputs;
    }

    if (mode & NLPGradient) {
        my_weights _w(neural_net, d->inputs, x, m, n);
        for (uint j = 1; j <= n; j++) {
            gx(j) = 0;
            for (uint i = 0; i < _w.h() ; i++) {
                double der = activation_function(_w.D(i), _w.lambda());
                gx(j) = gx(j) + _w.v(i) * _w.w(m + j - 1, i) * _w.lambda() * (1 - der * der);
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
                  Hx(j, k) = Hx(j, k) + -2.f * _w.lambda() * _w.w(m + k - 1, i) * Lambda_ij * der * (1 - der * der);
              }
            }
        }
        result = NLPHessian;
    }
}
