#ifndef BASECACLAAG_HPP
#define BASECACLAAG_HPP

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

class BaseCaclaAg : public arch::ARLAgent<> {
 public:
  BaseCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : ARLAgent<>(_nb_motors), nb_sensors(_nb_sensors), empty_action(0) {

  }

  virtual ~BaseCaclaAg() {
    delete vnn;
    delete ann;
    
    delete ann_testing;
    
    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool) {

    vector<double>* next_action ;
    if(learning)
      next_action = ann->computeOut(sensors);
    else
      // protect batch norm from testing data
      next_action = ann_testing->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached) {
        double nextV = vnn->computeOutVF(sensors, {});
        vtarget += gamma * nextV;
      }
      double lastv = vnn->computeOutVF(last_state, empty_action);

      vnn->learn(last_state, empty_action, vtarget);
      double delta = vtarget - lastv;

      if (delta > 0) { //increase this action
        if(plus_var_version) {
          uint number_update = ceil(delta / sqrt(delta_var));
          for(uint k=0; k < number_update; k++)
            ann->learn(last_state, *last_action);
        } else
          ann->learn(last_state, *last_action);
#ifndef NDEBUG
        //print gradient over actions
//         const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
//         auto ac_diff = actor_actions_blob->cpu_diff();
//         bib::Logger::PRINT_ELEMENTS(ac_diff, actor_actions_blob->count());
#endif
      }

      if(plus_var_version)
        delta_var = (1 - beta) * delta_var + beta * delta * delta;
    }

    if(learning) {
      if(gaussian_policy){
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
//     bib::Seed::setFixedSeedUTest();
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    plus_var_version        = pt->get<bool>("agent.plus_var_version");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    double alpha_v                 = pt->get<double>("agent.alpha_v");
    double alpha_a                 = pt->get<double>("agent.alpha_a");
    uint batch_norm_critic       = pt->get<uint>("agent.batch_norm_critic");
    uint batch_norm_actor        = pt->get<uint>("agent.batch_norm_actor");
    uint actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    uint hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    
    if(batch_norm_critic > 0 || batch_norm_actor > 0)
      LOG_WARNING("You want to use batch normalization but there is no batch.");
    
    beta = 0.001f;
    if(plus_var_version)
      beta                  = pt->get<uint>("agent.beta");
    delta_var = 1;
    uint kMinibatchSize = 1;

    vnn = new MLP(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, kMinibatchSize, -1, hidden_layer_type, batch_norm_critic);
    
    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, kMinibatchSize, hidden_layer_type, actor_output_layer_type, batch_norm_actor, true);
    
    ann_testing = new MLP(*ann, false, ::caffe::Phase::TEST);
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    
    if(!learning){
      double* weights = new double[ann->number_of_parameters(false)];
      ann->copyWeightsTo(weights, false);
      ann_testing->copyWeightsFrom(weights, false);
      delete[] weights;
    }
  }
  
  void save(const std::string& path, bool, bool) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << sum_weighted_reward << " " << std::setw(8) << std::fixed << std::setprecision(
          5) << vnn->error() << " " << noise ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << noise;
  }

 public:
  uint nb_sensors;
 private:
  std::vector<double> empty_action;

  double noise;
  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;

  double beta, delta_var;
  bool plus_var_version;
  bool gaussian_policy;

  std::shared_ptr<std::vector<double>> last_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  MLP* ann;
  MLP* vnn;
  
  MLP* ann_testing;
};

#endif

