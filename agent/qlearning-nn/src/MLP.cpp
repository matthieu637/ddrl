#include "MLP.hpp"

#include <vector>
#include "fann/doublefann.h"
#include "bib/Assert.hpp"
#include <bib/Utils.hpp>

#define activation_function(x, lambda) tanh(lambda * x)

ParallelOptimization::ParallelOptimization(const NN _neural_net, const std::vector<double>& _inputs,
    const std::vector<double>& _init_search,
    uint _size_motors, uint number_parra): neural_net(_neural_net), inputs(_inputs), init_search(_init_search),
  ndim(_size_motors), a(number_parra) {
  for (uint i = 0; i < number_parra; i++)
    a[i] = new std::vector<double>(_size_motors);

  cost = new std::vector<double>(number_parra);
}

ParallelOptimization::~ParallelOptimization() { //must be empty cause of tbb

}

void ParallelOptimization::free(uint expect_index) {
  delete cost;

  for (uint i = 0; i < a.size(); i++)
    if (i != expect_index)
      delete a[i];
}

// std::vector<uint> MLP::strat(16,0);

void ParallelOptimization::operator()(const tbb::blocked_range<uint>& r) const {

  ColumnVector lower(ndim), upper(ndim);

  for (uint i = 1; i <= ndim; i++) {
    lower(i) = -1.0f ;
    upper(i) = 1.0f ;
  }

  Constraint c1 = new BoundConstraint(ndim, lower, upper);
  CompoundConstraint constraints(c1);

  NN my_local_nn = fann_copy(neural_net);
  passdata d = {my_local_nn, inputs};

  for (uint thread = r.begin(); thread != r.end() ; thread++) {

    NLF2* nips = nullptr;
    if (init_search.size() == 0)
      nips = new NLF2(ndim, hs65, init_hs65_random, &constraints, &d);
    else
      nips = new UNLF2(ndim, hs65, init_search, &constraints, &d);
    nips->setIsExpensive(true); //only for line search : true -> backtrack

    uint strategy = thread % 5;
//tot=0; iter=20; while [ $iter -ne 0 ] ; do  ./unit-test >& /dev/null ; new_n=$? ; tot=`expr $tot + $new_n` ; echo $new_n ; iter=`expr $iter - 1` ; done ; echo "-$tot"
//         strategy = 1 + (thread % 4); // perf -> -3.5 (with 8 threads/solutions)
//         strategy = thread <= 1 ? thread + 1 : 4 ; // 1 2 4 4 4 4 4 4 //perf -> -8
//         strategy = 4; // perf -> -20
//         static std::vector<uint> full {1,2,4,4,4,1,3,4}; //perf -> -7
//         static std::vector<uint> full {1,2,4,4,4,1,3,1}; //perf -> -4
//         static std::vector<uint> full {1,2,4,4,0,1,3,1}; //perf -> -3.5
//         static std::vector<uint> full {0,1,2,4,0,1,3,4}; //perf -> -3

    //12 sols
//         static std::vector<uint> full {0,1,2,4, 0,1,3,4, 0,0,1,4}; //perf -> -3
//         static std::vector<uint> full {0,1,2,4, 0,4,3,4, 0,0,4,4}; //perf -> -5
    //20
    static std::vector<uint> full {0, 4, 1, 2, 3, 4, 0, 4, 1, 3, 4, 4, 0, 4, 0, 4, 0, 4, 0, 4};
    strategy = thread >= 20 ? full[8 + (thread % 12)] : full[thread] ;
//     strategy = full[thread % full.size()];

    OPTPP::OptimizeClass* pobjfcn;

    if (strategy == 0) {
//      #################################
//      Nonlinear Interior-Point Method Method with Line Search
//      for general constraints probleme, here we have only bound
      pobjfcn = new OptNIPS(nips);
//         objfcn.setMeritFcn(OPTPP::ArgaezTapia); //OPTPP::NormFmu ArgaezTapia don't change many things
//      seems to found bad solutions on unit test but works verry well with a populations of solutions
      static_cast<OptNIPS*>(pobjfcn)->setTRSize(1);
    }

    else if (strategy >= 1 && strategy <= 2) {
//      #################################
//      Bound constrained Newton with barrier Method with Line Search with a logarithmic barrier term.
      if (strategy == 2) {
        delete nips;
        nips = new NLF2(ndim, hs65, init_hs65_zero, &constraints, &d);
      }
      pobjfcn =  new OptBaNewton(nips);
//      Found good solution but don't terminate sometimes (infinite loop)
//      Better than NIPS of one solution (especially when starting from 0) but worst with populations of solutions

    } else if (strategy >= 3 && strategy <= 4) {
//      #################################
//      Bound constrained Newton with barrier Method with the selected search strategy
      pobjfcn =  new OptBCNewton(nips);
//      seems to found bad solutions on unit test + wrong parameters
//      Line Search : start from an extremum seems mandatory
//      Line Search problem -> backtrack: Initial search direction not a descent direction | cannot take a step
//      Line Search problem -> no backtrack: msearch direction not a descent direction | -nan
//      TrustRegion problem -> local maxima : cannot take a step (+100step)
      if (strategy == 3) {
        static_cast<OptBCNewton*>(pobjfcn)->setSearchStrategy(OPTPP::TrustRegion);
        static_cast<OptBCNewton*>(pobjfcn)->setTRSize(1);
      } else {
        static_cast<OptBCNewton*>(pobjfcn)->setSearchStrategy(OPTPP::LineSearch);
      }
    } else {
      LOG_DEBUG("never be called, not a strategy");
      exit(1);
    }

#ifdef NDEBUG
    pobjfcn->setOutputFile("/dev/null", 1);
#else
    std::string filename("optpp.log.");
    std::string filename2 = std::to_string(thread);
    pobjfcn->setOutputFile((filename + filename2).c_str(), 0);
#endif
    pobjfcn->setFcnTol(1.0e-06);
    pobjfcn->setMaxFeval(30);
    pobjfcn->setMaxIter(40);
    pobjfcn->setMaxStep(1. / 3.);
    pobjfcn->setMinStep(1e-7);
    pobjfcn->setLineSearchTol(1e-4);
    pobjfcn->setMaxBacktrackIter(10);

// objfcn.setDebug();
//         objfcn.printStatus("ici");
    pobjfcn->optimize();


    ColumnVector result = nips->getXc();
    //check solution are realisable
    if (!c1.amIFeasible(nips->getXc(), 1e-5)) {
      init_hs65_random(ndim, result);
      nips->setX(result);
      nips->evalF();
    }

    for (uint i = 0; i < a[thread]->size(); i++)
      a[thread]->at(i) = result(i + 1);



    cost->at(thread) = nips->getF();

    pobjfcn->cleanup();

    delete pobjfcn;

    delete nips;
  }

  fann_destroy(my_local_nn);
}

void init_hs65_zero(int ndim, ColumnVector& x) {
  for (uint i = 1; i <= (uint) ndim; i++)
    x(i) = 0;
}

void init_hs65_random(int ndim, ColumnVector& x) {
  for (uint i = 1; i <= (uint) ndim; i++)
    x(i) = bib::Utils::rand01() * 2. - 1.;
}

class my_weights {
  typedef struct fann_connection sfn;

 public:
  my_weights(NN neural_net, const std::vector<double>& sensors, const ColumnVector& x, uint _m, uint _n) : m(_m), n(_n) {
    _lambda = fann_get_activation_steepness(neural_net, 1, 0);

    unsigned int number_connection = fann_get_total_connections(neural_net);
    connections = reinterpret_cast<sfn*>(calloc(number_connection, sizeof(sfn)));

    fann_get_connection_array(neural_net, connections);

    uint number_layer = fann_get_num_layers(neural_net);
    layers = reinterpret_cast<uint*>(calloc(number_layer, sizeof(sfn)));

    fann_get_layer_array(neural_net, layers);
    ASSERT(number_layer == 3, number_layer);

    for (uint i = 0; i < h() * (m + n + 1); i++) {
      _ASSERT_EQ(connections[i].from_neuron, i % (m + n + 1));
      _ASSERT_EQ(connections[i].to_neuron, (m + n + 1) + i / (m + n + 1));
    }

    Ci.clear();
    Ci.resize(h());
    for (uint i = 0; i < h(); i++) {
      Ci[i] = 0;
      for (uint j = 0; j < m; j++)
        Ci[i] += sensors[j] * w(j, i);

      Ci[i] += connections[ i * (m + n + 1) + m + n].weight;
      _ASSERT_EQ(connections[ i * (m + n + 1) + m + n].from_neuron, m + n);
      _ASSERT_EQ(connections[ i * (m + n + 1) + m + n].to_neuron, m + n + 1 + i);
    }

    Di.clear();
    Di.resize(h());
    for (uint i = 0; i < h(); i++) {
      Di[i] = Ci[i];
      for (uint j = 1; j <= n; j++)
        Di[i] += x(j) * w(m + j - 1, i);
    }

    ASSERT(number_connection == h() * (m + n + 1) + (h() + 1), "");
  }

  ~my_weights() {
    free(connections);
    free(layers);
  }

  double v(uint i) const {
    sfn conn = connections[ h() * (m + n + 1) + i];

    _ASSERT_EQS(conn.from_neuron, (m + n + 1) + i, " i: " << i  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m + n + 1) + (h() + 1), " i: " << i  << " m " << m << " n " << n << " h " << h());
    return conn.weight;
  }

  double w(uint j, uint i) const {
    sfn conn = connections[ i * (m + n + 1) + j];

    _ASSERT_EQS(conn.from_neuron, j, " i: " << i << " j : " << j  << " m " << m << " n " << n << " h " << h());
    _ASSERT_EQS(conn.to_neuron, (m + n + 1) + i, " i: " << i << " j : " << j  << " m " << m << " n " << n << " h " << h());
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
//         LOG_DEBUG("my out " << out[0] << " for x = " << inputs[0] );
    delete[] inputs;
  }

  if (mode & NLPGradient || mode & NLPHessian) {
    my_weights _w(neural_net, d->inputs, x, m, n);
    if (mode & NLPGradient) {

      for (uint j = 1; j <= n; j++) {
        gx(j) = 0;
        for (uint i = 0; i < _w.h() ; i++) {
          double der = activation_function(_w.D(i), _w.lambda());
          gx(j) = gx(j) - _w.v(i) * _w.w(m + j - 1, i) * _w.lambda() * (1.0 - der * der);
        }
//       LOG_DEBUG("my grad " << j<< " " <<gx(j));
      }
      result = NLPGradient;
    }

    if (mode & NLPHessian) {
      for (uint j = 1; j <= n; j++) {
        for (uint k = 1; k <= n; k++) {
          Hx(j, k) = 0;
          for (uint i = 0; i < _w.h() ; i++) {
            double der = activation_function(_w.D(i), _w.lambda());
            double Lambda_ij = _w.v(i) * _w.w(m + j - 1, i) * _w.lambda();
            Hx(j, k) = Hx(j, k) - -2. * _w.lambda() * _w.w(m + k - 1, i) * Lambda_ij * der * (1.0 - der * der);
          }
//         LOG_DEBUG("my hess " << j<< " " << k << " "<< Hx(j, k));
        }
      }
      result = NLPHessian;
    }
  }
}
