
#ifndef FUSIONONOFFAG_HPP
#define FUSIONONOFFAG_HPP

#include "DeepQNAg.hpp"
#include "PenNFACAg.hpp"

class FusionOOAg : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  FusionOOAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentGPUProgOptions>(_nb_motors, _nb_sensors), 
        offpolicy_ag(_nb_motors, _nb_sensors), onpolicy_ag(_nb_motors, _nb_sensors) {

  }

  virtual ~FusionOOAg() {
    
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) override {
        
        if(! learning)
            return onpolicy_ag._run(reward, sensors, learning, goal_reached, false);
        
        const std::vector<double>& onpol_ac = onpolicy_ag._run(reward, sensors, learning, goal_reached, false);
        const std::vector<double>& offpol_ac = offpolicy_ag._run(reward, sensors, learning, goal_reached, false);
        
        if (offpolicy_ag.criticEval(sensors, onpol_ac) < offpolicy_ag.criticEval(sensors, offpol_ac)) {
            std::copy(offpol_ac.begin(), offpol_ac.end(), onpolicy_ag.last_action->begin());
            return offpol_ac;
        }
        
        std::copy(onpol_ac.begin(), onpol_ac.end(), offpolicy_ag.last_action->begin());
        return onpol_ac;
  }
  
  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    //hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    
    boost::property_tree::ptree* properties = new boost::property_tree::ptree;
        boost::property_tree::ini_parser::read_ini("config.off.ini", *properties);    
    offpolicy_ag.unique_invoke(properties, command_args, false);
    delete properties;
    
    onpolicy_ag.unique_invoke(pt, command_args, false);
  }

  void _start_episode(const std::vector<double>& sensors, bool learning) override {
    offpolicy_ag.start_episode(sensors, learning);
    onpolicy_ag.start_episode(sensors, learning);
  }

  void end_instance(bool learning) override {
    offpolicy_ag.end_instance(learning);
    onpolicy_ag.end_instance(learning);
  }
  
  void end_episode(bool learning) override {
    offpolicy_ag.end_episode(learning);
    onpolicy_ag.end_episode(learning);
  }

  void save(const std::string& path, bool save_best, bool) override {

  }

  void load(const std::string& path) override {

  }
  
  void save_run() override {
 
  }
  
  void load_previous_run() override {

  }
  
    double criticEval(const std::vector<double>&, const std::vector<double>&) override {
    LOG_INFO("not implemented");
    return 0;
  }
  
    arch::Policy<MLP>* getCopyCurrentPolicy() override {
//         return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
    return nullptr;
  }

 protected:
  void _display(std::ostream& out) const override {
//     out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward 
//     #ifndef NDEBUG
//     << " " << std::setw(8) << std::fixed << std::setprecision(5) << noise 
//     << " " << trajectory.size() 
//     << " " << ann->weight_l1_norm() 
//     << " " << std::fixed << std::setprecision(7) << qnn->error() 
//     << " " << qnn->weight_l1_norm()
//     #endif
    ;
  }

  void _dump(std::ostream& out) const override {
//     out << std::setw(25) << std::fixed << std::setprecision(22) <<
//         sum_weighted_reward << " " << std::setw(8) << std::fixed <<
//         std::setprecision(5) << trajectory.size() ;
  }

 private:
  DeepQNAg<MLP> offpolicy_ag;
  OfflineCaclaAg<MLP> onpolicy_ag;
};

#endif

