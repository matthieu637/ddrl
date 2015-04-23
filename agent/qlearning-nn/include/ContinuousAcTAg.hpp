#ifndef CONTINUOUSACTAG_HPP
#define CONTINUOUSACTAG_HPP

#include <vector>
#include <string>

#include "arch/AAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include "MLP.hpp"

class ContinuousAcTAg : public arch::AAgent<> {
 public:
  ContinuousAcTAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  ~ContinuousAcTAg() {
    delete nn;
  }
  
    const std::vector<float>& run(float reward, const std::vector<float>& sensors,
                                bool learning, bool goal_reached) override {
                                  
    if(reward >= 1.f){
        reward = 100;
        
        uint keeped = 2000-internal_time;
        reward = 100*log2(keeped+2);
    }
    internal_time ++;
    
    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;
    
    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;
    
    time_for_ac--;
    if(time_for_ac == 0){
      const std::vector<float>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
      time_for_ac = bib::Utils::transform(next_action[nb_motors], -1.,1., min_ac_time, max_ac_time);
      
      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];
      
      weighted_reward = 0;
      pow_gamma = 1.f;
    }
    
    return returned_ac;
  }

  const std::vector<float>& _run(float reward, const std::vector<float>& sensors,
                                bool learning, bool goal_reached) {
    vector<float>* next_action = nullptr;
    if(init_old_ac && last_action.get() != nullptr)
      next_action = nn->optimized(sensors, *last_action);
    else
      next_action = nn->optimized(sensors);

    if (last_action.get() != nullptr && learning) {  // Update Q
      double nextQ = nn->computeOut(sensors, *next_action);
      if (!goal_reached){
        if(aware_ac_time)
          nn->learn(last_state, *last_action, reward + pow(gamma, bib::Utils::transform(last_action->at(last_action->size()-1),-1.,1., min_ac_time, max_ac_time) ) * nextQ);
        else
          nn->learn(last_state, *last_action, reward + gamma * nextQ);
      }
      else{
        nn->learn(last_state, *last_action, reward);
      }
    }

    if (bib::Utils::rand01() < alpha) {
      for (uint i = 0; i < next_action->size(); i++)
        next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
    }
    last_action.reset(next_action);

    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }

  void _unique_invoke(boost::property_tree::ptree*, boost::program_options::variables_map*) override {
//         epsilon             = pt->get<float>("agent.epsilon");
//         gamma               = pt->get<float>("agent.gamma");
//         alpha               = pt->get<float>("agent.alpha");
//         hidden_unit         = pt->get<int>("agent.hidden_unit");
// //     rlparam->activation          = pt->get<std::string>("agent.activation_function_hidden");
// //     rlparam->activation_stepness = pt->get<float>("agent.activation_steepness_hidden");
// //
// //     rlparam->repeat_replay = pt->get<int>("agent.replay");
// //
// //     int action_per_motor   = pt->get<int>("agent.action_per_motor");
// //
// //     sml::ActionFactory::getInstance()->gridAction(nb_motors, action_per_motor);
// //     actions = new sml::list_tlaction(sml::ActionFactory::getInstance()->getActions());
// //
// //     act_templ = new sml::ActionTemplate( {"effectors"}, {sml::ActionFactory::getInstance()->getActionsNumber()});
// //     ainit = new sml::DAction(act_templ, {0});
// //     algo = new sml::QLearning<EnvState>(act_templ, *rlparam, nb_sensors);
    hidden_unit=35;
//     gamma = 0.999; // < 0.99  => gamma ^ 2000 = 0 && gamma != 1 -> better to reach the goal at the very end
    gamma = 1.0d;
    //check 0,0099×((1−0.95^1999)÷(1−0.95)) 
    //r_max_no_goal×((1−gamma^1999)÷(1−gamma)) < r_max_goal * gamma^2000 && gamma^2000 != 0
    alpha = 0.01;
    epsilon = 0.15;
    
    min_ac_time = 5;
    max_ac_time = 15;
    
    aware_ac_time = false;
    init_old_ac = false;
  
    nn = new MLP(nb_sensors + nb_motors + 1, hidden_unit, nb_sensors, alpha);
  }

  void start_episode(const std::vector<float>& sensors) override {
//     EnvState s(new std::vector<float>(sensors));
//     algo->startEpisode(s, *ainit);
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);
    
    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;
    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
  }

  void end_episode() override {

  }

  void save(const std::string& path) override {
      nn->save(path);
  }

  void load(const std::string& path) override {
      nn->load(path);
  }

 protected:
  void _display(std::ostream& out) const override {
      out << sum_weighted_reward ;
  }

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;
  
  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;
  
  uint min_ac_time;
  uint max_ac_time;
  
  uint internal_time;
  
  bool aware_ac_time;
  bool init_old_ac;

  double epsilon, alpha, gamma;
  uint hidden_unit;

  std::shared_ptr<std::vector<float>> last_action;
  std::vector<float> last_state;
  
  std::vector<float> returned_ac;

  MLP* nn;
};

#endif
