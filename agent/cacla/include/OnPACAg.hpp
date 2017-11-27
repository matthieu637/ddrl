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

class OnPACAg : public arch::ARLAgent<> {
 public:
  OnPACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : ARLAgent<>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~OnPACAg() {
    delete qnn;
    delete ann;

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
    } else if(learning && actor_output_layer_type == 0) {
      shrink_actions(next_action);
    }

    if (last_action.get() != nullptr && learning) {  // Update Q

      double qtarget = reward;
      double qtarget_pi = reward;
      if (!goal_reached) {
        double nextQ = qnn->computeOutVF(sensors, *next_action);
        qtarget += gamma * nextQ;
        if(stochastic_gradient) {
          vector<double>* ac = ann->computeOut(last_state);
          double vs = 0;
          for(int j=0;j<10;j++){
            vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*ac, noise);
            vs += qnn->computeOutVF(last_state, *randomized_action);
            delete randomized_action;
          }
          qtarget_pi += qtarget - vs/10.f;
          delete ac;
        }
      }
//       double lastv = qnn->computeOutVF(last_state, *last_action);
//       double delta = qtarget - lastv;

      if(!delay_q_update)
        qnn->learn(last_state, *last_action, qtarget);

      //update actor with Q grad error
      if(proba_actor_update < 0.f || 
        bib::Utils::rand01() >= proba_actor_update_current ){
        ann->ZeroGradParameters();

        auto actions_outputs = ann->computeOut(last_state);
        if(stochastic_gradient){
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          auto ac_diff = actor_actions_blob->mutable_cpu_diff();
          for(int i=0; i<actor_actions_blob->count(); i++)
            ac_diff[i] = -qtarget_pi*(last_action->at(i)-actions_outputs->at(i));
          ann->actor_backward();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
        } else {
          qnn->ZeroGradParameters();
          qnn->computeOutVF(last_state, *actions_outputs);
          const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
          double* q_values_diff = q_values_blob->mutable_cpu_diff();
          q_values_diff[q_values_blob->offset(0,0,0,0)] = -1.0f;
          qnn->critic_backward();
          const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);

          // Transfer input-level diffs from Critic to Actor
          const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
          actor_actions_blob->ShareDiff(*critic_action_blob);
          ann->actor_backward();
          ann->getSolver()->ApplyUpdate();
          ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
        }
        delete actions_outputs;
      }

      if(delay_q_update)
        qnn->learn(last_state, *last_action, qtarget);
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
    actor_output_layer_type       = pt->get<uint>("agent.actor_output_layer_type");
    uint hidden_layer_type        = pt->get<uint>("agent.hidden_layer_type");
    on_policy                     = pt->get<bool>("agent.on_policy");
    delay_q_update                = pt->get<bool>("agent.delay_q_update");
    proba_actor_update            = pt->get<double>("agent.proba_actor_update");
    stochastic_gradient           = pt->get<bool>("agent.stochastic_gradient");
    uint kMinibatchSize = 1;
    proba_actor_update_current = 1;
    
    if(batch_norm_critic > 0 || batch_norm_actor > 0)
      LOG_WARNING("You want to use batch normalization but there is no batch.");

    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_v, alpha_v, kMinibatchSize, -1, hidden_layer_type,
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
  
  void end_episode(bool learning) override {
    if(learning)
      proba_actor_update_current = proba_actor_update_current * proba_actor_update;
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
  void shrink_actions(vector<double>* next_action) {
    for(uint i=0; i < nb_motors ; i++)
      if(next_action->at(i) > 1.f)
        next_action->at(i)=1.f;
      else if(next_action->at(i) < -1.f)
        next_action->at(i)=-1.f;
  }

  uint nb_sensors;

  double noise;
  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;

  uint actor_output_layer_type;
  bool gaussian_policy, on_policy, delay_q_update, stochastic_gradient;
  double proba_actor_update;
  double proba_actor_update_current;

  std::shared_ptr<std::vector<double>> last_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  MLP* ann;
  MLP* qnn;

  MLP* ann_testing;
};

#endif


