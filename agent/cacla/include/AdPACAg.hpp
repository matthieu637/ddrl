#ifndef ONPACAG_HPP
#define ONPACAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/ARLAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "nn/MLP.hpp"

class AdPACAg : public arch::ARLAgent<> {
 public:
  AdPACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : ARLAgent<>(_nb_motors), nb_sensors(_nb_sensors), empty_action(0) {

  }

  virtual ~AdPACAg() {
    delete qnn;
    delete ann;
    delete adnn;

    delete ann_testing;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) override {

    vector<double>* next_action ;
    if(learning)
      next_action = ann->computeOut(sensors);
    else
      // protect batch norm from testing data
      next_action = ann_testing->computeOut(sensors);

    if(learning && on_policy) {
      if(gaussian_policy) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else {
        if(bib::Utils::rand01() < noise)
          for (uint i = 0; i < next_action->size(); i++)
            next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
      }
    }

    if (last_action.get() != nullptr && learning) {  // Update Q

      double qtarget = reward;
      if (!goal_reached) {
        double nextQ = qnn->computeOutVF(sensors, *next_action);
        qtarget += gamma * nextQ;
      }
      //       double lastv = qnn->computeOutVF(last_state, *last_action);
      //       double delta = qtarget - lastv;
      qnn->learn(last_state, *last_action, qtarget);
      
      
      auto actions_outputs = ann->computeOut(last_state);
      double vtarget = qnn->computeOutVF(last_state, *actions_outputs);
      adnn->learn(last_state, *last_action, qtarget - vtarget);

      //update actor with Q grad error
      adnn->ZeroGradParameters();
      ann->ZeroGradParameters();

      adnn->computeOutVF(last_state, *actions_outputs);

      const auto q_values_blob = adnn->getNN()->blob_by_name(MLP::q_values_blob_name);
      double* q_values_diff = q_values_blob->mutable_cpu_diff();
      q_values_diff[q_values_blob->offset(0,0,0,0)] = -1.0f;
      adnn->critic_backward();
      const auto critic_action_blob = adnn->getNN()->blob_by_name(MLP::actions_blob_name);

      // Transfer input-level diffs from Critic to Actor
      const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
      actor_actions_blob->ShareDiff(*critic_action_blob);
      ann->actor_backward();
      ann->getSolver()->ApplyUpdate();
      ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);

      delete actions_outputs;
    }

    if(learning && !on_policy) {
      if(gaussian_policy) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else {
        if(bib::Utils::rand01() < noise)
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
    hidden_unit_v                 = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a                 = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                         = pt->get<double>("agent.noise");
    gaussian_policy               = pt->get<bool>("agent.gaussian_policy");
    double alpha_v                = pt->get<double>("agent.alpha_v");
    double alpha_a                = pt->get<double>("agent.alpha_a");
    uint batch_norm_critic        = pt->get<uint>("agent.batch_norm_critic");
    uint batch_norm_actor         = pt->get<uint>("agent.batch_norm_actor");
    uint actor_output_layer_type  = pt->get<uint>("agent.actor_output_layer_type");
    uint hidden_layer_type        = pt->get<uint>("agent.hidden_layer_type");
    on_policy                     = pt->get<bool>("agent.on_policy");
    uint kMinibatchSize = 1;
    
    if(batch_norm_critic > 0 || batch_norm_actor > 0)
      LOG_WARNING("You want to use batch normalization but there is no batch.");
    
    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_v, alpha_v, kMinibatchSize, -1, hidden_layer_type,
                  batch_norm_critic);
    
    adnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_v, alpha_v, kMinibatchSize, -1, hidden_layer_type,
                  batch_norm_critic);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, hidden_layer_type,
                  actor_output_layer_type, batch_norm_actor, false);

    ann_testing = new MLP(*ann, false, ::caffe::Phase::TEST);
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;

    if(!learning) {
      double* weights = new double[ann->number_of_parameters(false)];
      ann->copyWeightsTo(weights, false);
      ann_testing->copyWeightsFrom(weights, false);
      delete[] weights;
    }
  }

  void save(const std::string& path, bool, bool) override {
    ann->save(path+".actor");
    qnn->save(path+".critic");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    qnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << sum_weighted_reward << " " << std::setw(8) << std::fixed << std::setprecision(
          5) << qnn->error() << " " << noise ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << qnn->error() << " " << noise;
  }

 private:
  uint nb_sensors;

  double noise;
  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;

  bool gaussian_policy, on_policy;

  std::vector<double> empty_action;
  std::shared_ptr<std::vector<double>> last_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  MLP* ann;
  MLP* qnn;
  MLP* adnn;

  MLP* ann_testing;
};

#endif


