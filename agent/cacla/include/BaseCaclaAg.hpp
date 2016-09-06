#ifndef BASECACLAAG_HPP
#define BASECACLAAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/ARLAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "MLP.hpp"

class BaseCaclaAg : public arch::ARLAgent<> {
 public:
  BaseCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : ARLAgent<>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~BaseCaclaAg() {
    delete vnn;
    delete ann;
    
    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool finished) {

    vector<double>* next_action = ann->computeOut(sensors);
    

    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached && !finished) {
        double nextV = vnn->computeOutVF(sensors, {});
        vtarget += gamma * nextV;
      }
      double lastv = vnn->computeOutVF(last_state, *next_action);

      vnn->learn(last_state, {}, vtarget);
      double delta = vtarget - lastv;

      if (delta > 0) { //increase this action
        if(plus_var_version) {
          uint number_update = ceil(delta / sqrt(delta_var));
          for(uint k=0; k < number_update; k++)
            ann->learn(last_state, *last_action);
        } else
          ann->learn(last_state, *last_action);
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
    hidden_unit_v       = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a       = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    alpha_v             = pt->get<double>("agent.alpha_v");
    alpha_a             = pt->get<double>("agent.alpha_a");
    noise               = pt->get<double>("agent.noise");
    plus_var_version    = pt->get<bool>("agent.plus_var_version");
    gaussian_policy     = pt->get<bool>("agent.gaussian_policy");
    lecun_activation    = pt->get<bool>("agent.lecun_activation");
    bool last_activation_linear = pt->get<bool>("agent.last_activation_linear");
    
    beta = 0.001;
    delta_var = 1;
    
    vnn = new MLP(nb_sensors, *hidden_unit_v, nb_sensors, alpha_v, lecun_activation);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, lecun_activation);
    fann_set_training_algorithm(ann->getNeuralNet(), FANN_TRAIN_INCREMENTAL);
    
    fann_set_learning_rate(ann->getNeuralNet(), alpha_a / fann_get_total_connections(ann->getNeuralNet()));
    fann_set_learning_rate(vnn->getNeuralNet(), alpha_v / fann_get_total_connections(vnn->getNeuralNet()));
    if(last_activation_linear)
      fann_set_activation_function_output(ann->getNeuralNet(), FANN_LINEAR);
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;

    fann_reset_MSE(vnn->getNeuralNet());
  }

  void end_episode() override {
//      write_actionf_file("ac_func.data");
//      write_valuef_file("v_after.data");
//     ann->print();
//     LOG_FILE("policy_exploration", ann->hash());
  }
  
   void write_actionf_file(const std::string& file){
      
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
//         out << ann->computeOut(x)[0];
        LOG_ERROR("todo");
        out << std::endl;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 6);
      out.close();
/*
      close all; clear all; 
      X=load("ac_func.data");
      tmp_ = X(:,2); X(:,2) = X(:,3); X(:,3) = tmp_;
      key = X(:, 1:2);
      for i=1:size(key, 1)
       subkey = find(sum(X(:, 1:2) == key(i,:), 2) == 2);
       data(end+1, :) = [key(i, :) mean(X(subkey, end))];
      endfor
      [xx,yy] = meshgrid (linspace (-1,1,300));
      griddata(data(:,1), data(:,2), data(:,3), xx, yy, "linear"); xlabel('theta_1'); ylabel('theta_2');
*/    
   }
  
  void write_valuef_file(const std::string& file){
      
      std::ofstream out;
      out.open(file, std::ofstream::out);
    
      auto iter = [&](const std::vector<double>& x) {
        for(uint i=0;i < x.size();i++)
          out << x[i] << " ";
        out << vnn->computeOutVF(x, {});
        out << std::endl;
      };
      
      bib::Combinaison::continuous<>(iter, nb_sensors, -1, 1, 6);
      out.close();
/*
      function doit()
      close all; clear all; 
      X=load("v_after.data");
      X=load("v_before.data"); 
      tmp_ = X(:,2); X(:,2) = X(:,3); X(:,3) = tmp_;
      key = X(:, 1:2);
      for i=1:size(key, 1)
       subkey = find(sum(X(:, 1:2) == key(i,:), 2) == 2);
       data(end+1, :) = [key(i, :) mean(X(subkey, end))];
      endfor
      [xx,yy] = meshgrid (linspace (-1,1,300));
      griddata(data(:,1), data(:,2), data(:,3), xx, yy, "linear"); xlabel('theta_1'); ylabel('theta_2');
      endfunction
*/    
  }

  void save(const std::string& path) override {
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

 private:
  uint nb_sensors;

  double alpha_v, alpha_a;
  double noise;
  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;

  double beta, delta_var;
  bool plus_var_version;
  bool gaussian_policy;
  bool lecun_activation;

  std::shared_ptr<std::vector<double>> last_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  MLP* ann;
  MLP* vnn;
};

#endif

