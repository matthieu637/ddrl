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

#include "arch/AAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include <bib/Combinaison.hpp>
#include "MLP.hpp"
#include "LinMLP.hpp"
#include "Trajectory.hpp"

class BaseCaclaAg : public arch::AAgent<> {
 public:
  BaseCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors), obs(nb_sensors) {

  }

  virtual ~BaseCaclaAg() {
    delete vnn;
    delete ann;
  }

  const std::vector<double>& run(double r, const std::vector<double>& sensors,
                                bool learning, bool goal_reached) override {

    const std::vector<double>* sin;
    if(standard_score){
      std::vector<double> s(nb_sensors);
      obs.transform(s, sensors);
      sin = &s;
    } else
      sin = &sensors;

//     if(r >= 1.)
//       noise = 0.00f;

    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<double>& next_action = _run(weighted_reward, *sin, learning, goal_reached);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }
    
//     LOG_FILE("test_ech.data", sensors[0] << " "<< sensors[1]<< " "<< sensors[2]<< " "<< sensors[3]<< " "<< sensors[4]<< " " << sensors.size());
//     LOG_FILE("test_echa.data", returned_ac[0] );
//     LOG_FILE("test_ech.data", s[0] << " "<< s[1]<< " "<< s[2]<< " "<< s[3]<< " "<< s[4]<< " " << s.size());
/*
    clear all; close all; X=load('test_ech.data' );
    for i=1:size(X,2)
      figure; plot(X(:,i));
    endfor
*/
    return returned_ac;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached) {

    vector<double>* next_action = ann->computeOut(sensors);
    
//     LOG_DEBUG(next_action->at(0));

    if (last_action.get() != nullptr && learning) {  // Update Q

      double vtarget = reward;
      if (!goal_reached) {
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
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
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

    gamma               = pt->get<double>("agent.gamma");
    alpha_v             = pt->get<double>("agent.alpha_v");
    alpha_a             = pt->get<double>("agent.alpha_a");
    noise               = pt->get<double>("agent.noise");
    hidden_unit_v       = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a       = pt->get<int>("agent.hidden_unit_a");
    decision_each       = pt->get<int>("agent.decision_each");
    plus_var_version    = pt->get<bool>("agent.plus_var_version");
    standard_score      = pt->get<bool>("agent.standard_score");
    gaussian_policy     = pt->get<bool>("agent.gaussian_policy");
    lecun_activation    = pt->get<bool>("agent.lecun_activation");
    
    if(!gaussian_policy)
      noise=0.1;
    
    beta = 0.001;
    delta_var = 1;
    
    if(hidden_unit_v == 0){
      vnn = new LinMLP(nb_sensors, 1, alpha_v, lecun_activation);
      fann_set_activation_function_output(vnn->getNeuralNet(), FANN_LINEAR);
    } else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, alpha_v, lecun_activation);

    if(hidden_unit_a == 0){
      ann = new LinMLP(nb_sensors , nb_motors, alpha_a, lecun_activation);
    }
     else {
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
      fann_set_training_algorithm(ann->getNeuralNet(), FANN_TRAIN_INCREMENTAL);
    }
    
    fann_set_learning_rate(ann->getNeuralNet(), alpha_a / fann_get_total_connections(ann->getNeuralNet()));
    fann_set_learning_rate(vnn->getNeuralNet(), alpha_v / fann_get_total_connections(vnn->getNeuralNet()));
  }

  void start_episode(const std::vector<double>& sensors) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;

    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;

    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
    
//     noise *= 0.999;
//     alpha_a *= 0.99999f;
//     alpha_v *= 0.99999f;

    fann_reset_MSE(vnn->getNeuralNet());
  }

//     double sign(double x){
//         if (x>0)
//             return 1;
//         else if(x<0)
//             return -1;
//         return 0;
//     }

  void end_episode() override {
//         double last_E = last_last_sum_weighted_reward - last_sum_weighted_reward;
//         double E = last_sum_weighted_reward - sum_weighted_reward;
//
//         if(last_E * E > 0 ){
//             rprop_factor = std::min(rprop_factor * 1.2, 0.1);
//         } else if(last_E * E < 0) {
//             rprop_factor = std::max(rprop_factor * 0.5, 0.01);
// //             E = 0;
//         }
//
// //         noise = noise - sign(E) * rprop_factor;
//
//         if(E > 0)
//             noise = noise - rprop_factor;
//         else
//             noise = noise + rprop_factor;
    
//      write_actionf_file("ac_func.data");
//      write_valuef_file("v_after.data");
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
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;

  uint internal_time, decision_each;

  double alpha_v, gamma, alpha_a;
  double noise;
  uint hidden_unit_v, hidden_unit_a;

  double beta, delta_var;
  bool plus_var_version;
  bool standard_score;
  bool gaussian_policy;
  bool lecun_activation;

  std::shared_ptr<std::vector<double>> last_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  MLP* ann;
  MLP* vnn;
  
  Observer obs;
};

#endif

