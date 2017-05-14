#ifndef RANDOMNNAG_HPP
#define RANDOMNNAG_HPP

#include <vector>
#include <string>

#include "arch/ARLAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include "nn/MLP.hpp"

template<typename NN = MLP>
class RandomNNAg : public arch::ARLAgent<arch::AgentProgOptions> {
 public:
  RandomNNAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::ARLAgent<arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~RandomNNAg() {
    delete ann;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double, const std::vector<double>& sensors,
                                  bool, bool, bool) {

    vector<double>* next_action = ann->computeOut(sensors);
    last_action.reset(next_action);
    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_a               = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    uint actor_hidden_layer_type     = pt->get<uint>("agent.actor_hidden_layer_type");
    uint actor_output_layer_type     = pt->get<uint>("agent.actor_output_layer_type");
    uint batch_norm                  = pt->get<uint>("agent.batch_norm");

    ann = new NN(nb_sensors, *hidden_unit_a, nb_motors, 0.1, 1, actor_hidden_layer_type, 
                 actor_output_layer_type, batch_norm);
  }

  void _start_episode(const std::vector<double>& , bool) override {
    const uint dimension = ann->number_of_parameters(false);
    double* startx  = new double[dimension];
    for(uint j=0; j< dimension; j++)
      startx[j] = bib::Utils::randin(-10, 10);
    ann->copyWeightsFrom(startx, false);
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

  void _dump(std::ostream& out) const override {
    out << std::setw(8) << std::fixed << std::setprecision(5) << sum_weighted_reward;
  }

 private:
  //initilized by constructor
  uint nb_sensors;

  //initialized by invoke
  std::vector<uint>* hidden_unit_a;
  MLP* ann;
  
  std::shared_ptr<std::vector<double>> last_action;
};

#endif


