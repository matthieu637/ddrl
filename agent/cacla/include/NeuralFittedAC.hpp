
#ifndef NEURALFITTEDAC_HPP
#define NEURALFITTEDAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "cmaes_interface.h"

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "MLP.hpp"
#include "kde.hpp"
#include "kdtree++/kdtree.hpp"
#include "bib/Combinaison.hpp"

#define DOUBLE_COMPARE_PRECISION 1e-9

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  double p0;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(pure_a);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

  //Used to store all sample into a tree, might be stochastic
  //only pure_a is negligate
  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return s[i] < b.s[i];
    }

    for (uint i = 0; i < a.size(); i++) {
      if(fabs(a[i] - b.a[i])>=DOUBLE_COMPARE_PRECISION)
        return a[i] < b.a[i];
    }

    for (uint i = 0; i < next_s.size(); i++) {
      if(fabs(next_s[i] - b.next_s[i])>=DOUBLE_COMPARE_PRECISION)
        return next_s[i] < b.next_s[i];
    }

    if(fabs(r - b.r)>=DOUBLE_COMPARE_PRECISION)
      return r < b.r;

    return goal_reached < b.goal_reached;
  }

  typedef double value_type;

  inline double operator[](size_t const N) const {
    return s[N];
  }

  bool same_state(const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return false;
    }

    return true;
  }

} sample;

class NeuralFittedAC : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;

  NeuralFittedAC(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~NeuralFittedAC() {
    delete kdtree_s;

    delete vnn;
    delete ann;
    
    delete hidden_unit_q;
    delete hidden_unit_a;
  }

  inline vector<double>* policy(const std::vector<double>& sensors) {
    vector<double>* next_action = ann->computeOut(sensors);
//     shrink_actions(next_action);
    return next_action;


//     return vnn->optimizedBruteForce(sensors,0.01);
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool last) {

    vector<double>* next_action = policy(sensors);

    if (last_action.get() != nullptr && learning) {
      double p0 = 1.f;
      for(uint i=0; i < nb_motors; i++)
        p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));

      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || last, p0};
      trajectory.push_back(sa);
      kdtree_s->insert(sa);
      proba_s.add_data(last_state);
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise) { //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    converge_precision      = pt->get<double>("agent.converge_precision");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");

    vnn = new MLP(nb_sensors + nb_motors, *hidden_unit_q, nb_sensors, 0.0, lecun_activation);

//     fann_set_learning_l2_norm(vnn->getNeuralNet(), 0.001);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, lecun_activation);
//  needed for inverting grad:
//     fann_set_activation_function_output(ann->getNeuralNet(), FANN_LINEAR);
//     fann_set_learning_l2_norm(ann->getNeuralNet(), 0.001);//implies some DEPROVE but stability weight sum

    sigma.resize(nb_sensors);
    for (uint i = 0; i < nb_sensors ; i++) {
      sigma[i] = bib::Utils::rand01();
    }

    kdtree_s = new kdtree_sample(nb_sensors);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    //trajectory.clear();

    fann_reset_MSE(vnn->getNeuralNet());
    learning = _learning;
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, NeuralFittedAC* _ptr) : vtraj(_vtraj), ptr(_ptr),
      actions(_vtraj.size()) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors+ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];

//         actions[n] = ptr->ann->computeOut(sm.next_s);
        actions[n] = ptr->policy(sm.next_s);
      }
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);

      for (uint n = 0; n < vtraj.size(); n++)
        delete actions[n];
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          std::vector<double> * next_action = actions[n];
          double nextQA = MLP::computeOutVF(local_nn, sm.next_s, *next_action);
          delta += ptr->gamma * nextQA;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    NeuralFittedAC* ptr;
    std::vector<std::vector<double> *> actions;
  };


  void computePTheta(vector< sample >& vtraj, double *ptheta) {
    uint i=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      sample sm = *it;
//       vector<double>* next_action = ann->computeOut(sm.s);
      vector<double>* next_action = policy(sm.s);

      double p0 = 1.f;
      for(uint i=0; i < nb_motors; i++)
        p0 *= exp(-(next_action->at(i)-sm.a[i])*(next_action->at(i)-sm.a[i])/(2.f*noise*noise));

      ptheta[i] = p0;
      i++;
      delete next_action;
    }
  }


  void update_critic() {
    if (trajectory.size() > 0) {
      //remove trace of old policy

      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
      //for testing perf - do it before importance sample computation
      std::random_shuffle(vtraj.begin(), vtraj.end());

      double *importance_sample = new double [trajectory.size()];

      double * ptheta = new double [trajectory.size()];
      computePTheta(vtraj, ptheta);
      //compute 1/u(x)
      uint i=0;
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        importance_sample[i] = ptheta[i] / it->p0;
        if(importance_sample[i] > 1.f)
          importance_sample[i] = 1.f;
        i++;
      }

      delete[] ptheta;

      //overlearning test
      uint nb_app_sample = (uint) (((float) vtraj.size()) * (100.f/100.f));
      uint nb_test_sample = vtraj.size() - nb_app_sample;
      std::vector<sample> vtraj_app(nb_app_sample);
      std::copy(vtraj.begin(), vtraj.begin() + nb_app_sample, vtraj_app.begin());
      std::vector<sample> vtraj_test(nb_test_sample);
      std::copy(vtraj.begin() + nb_app_sample, vtraj.end(), vtraj_test.begin());

      ParraVtoVNext dq(vtraj_app, this);
      ParraVtoVNext dq_test(vtraj_test, this);

      auto iter = [&]() {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_app.size()), dq);
//           dq(tbb::blocked_range<size_t>(0, vtraj.size()));

        if(vnn_from_scratch)
          fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);

//             vnn->learn_stoch_lw(dq.data, importance_sample, 10000, 0, converge_precision);
        vnn->learn_stoch(dq.data, 300, 0, converge_precision);
      };

      auto eval = [&]() {
        double app_error = fann_get_MSE(vnn->getNeuralNet());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_test.size()), dq_test);

        double* out;
        double test_error = 0.f;
        for(uint j = 0; j < dq_test.data->num_data; j++) {
          out = fann_run(vnn->getNeuralNet(), dq_test.data->input[j]);

          double l = (dq_test.data->output[j][0] - out[0]);
//             test_error += l*l;

          test_error += l*l * importance_sample[j + nb_app_sample];
        }


//           LOG_DEBUG(app_error << " " << test_error);

        if(dq_test.data->num_data == 0)
          return app_error;

        test_error = sqrt(test_error) / vtraj_test.size();
        return test_error;
//           return app_error;
      };

      if(determinist_vnn_update) {
          iter();
//         bib::Converger::determinist<>(iter, eval, number_fitted_iteration, converge_precision, 0, "deter_critic");
      } else {
        NN best_nn = nullptr;
        auto save_best = [&]() {
          if(best_nn != nullptr)
            fann_destroy(best_nn);
          best_nn = fann_copy(vnn->getNeuralNet());
        };

        bib::Converger::min_stochastic<>(iter, eval, save_best, 30, converge_precision, 1, 10, "stoch_crtic");
        vnn->copy(best_nn);
        fann_destroy(best_nn);
      }

      dq.free();
      dq_test.free();
      delete[] importance_sample;
//         LOG_DEBUG("critic updated");
    }
  }

  void update_actor_nfqca() {
    if(trajectory.size() > 0) {
      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
      std::random_shuffle(vtraj.begin(), vtraj.end());

      uint nb_app_sample = (uint) (((float) trajectory.size()) * (100.f/100.f));
      struct fann_train_data* data = fann_create_train(nb_app_sample, nb_sensors, nb_motors);

      uint n=0;
      for(uint i = 0; i < nb_app_sample; i++) {
        sample sm = vtraj[i];

        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];

        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = 0.f;//sm.a[i]; //don't care

        n++;
      }

      datann_derivative d = {vnn, (int)nb_sensors, (int)nb_motors};

      auto iter = [&]() {
//         fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn_inverting, &d);
        fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
      };

      auto eval = [&]() {
        //compute weight sum
//         return ann->weight_l1_norm();
//         return fitfun_sum_overtraj();
      };

//       bib::Converger::determinist<>(iter, eval, 10, 0.0001, 10, "actor grad");


      //alone update
      for(uint i=0; i<25; i++)
        iter();

      fann_destroy_train(data);

    }
  }

  void end_episode() override {
    if(!learning)
      return;
    
    while(trajectory.size() > 5000)
      trajectory.pop_front();

    auto iter = [&]() {
      update_critic();
//       LOG_DEBUG(ann->weight_l1_norm() << " " << fann_get_MSE(vnn->getNeuralNet()) <<" "  <<
//                 vnn->weight_l1_norm());

      update_actor_nfqca();
    };

    int k = 0;
    for(k =0; k<10; k++)
      iter();
  }

  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
    return vnn->computeOutVF(perceptions, actions);
  }

  arch::Policy<MLP>* getCopyCurrentPolicy() override {
    return new arch::Policy<MLP>(
             new MLP(*ann) ,
             gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY,
             noise,
             decision_each);
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
    bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
//     ann->load(path+".actor");
//     vnn->load(path+".critic");
    ann->load(path);
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size() ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }


 private:
  uint nb_sensors;

  double noise, converge_precision;
  bool gaussian_policy, vnn_from_scratch, lecun_activation,
       determinist_vnn_update;
  uint number_fitted_iteration;

  bool learning;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::vector<double> sigma;

  std::deque<sample> trajectory;
  KDE proba_s;

  struct L1_distance {
    typedef double distance_type;

    double operator() (const double& __a, const double& __b, const size_t) const {
      double d = fabs(__a - __b);
      return d;
    }
  };

  struct L2_distance {
    typedef double distance_type;

    double operator() (const double& __a, const double& __b, const size_t) const {
      double d = (__a - __b);
      d = sqrt(d*d);
      return d;
    }
  };

  typedef KDTree::KDTree<sample, KDTree::_Bracket_accessor<sample>, L2_distance> kdtree_sample;
  kdtree_sample* kdtree_s;

  MLP* ann;
  MLP* vnn;
  
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
};

#endif


