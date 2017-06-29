#ifndef ADENFAC_HPP
#define ADENFAC_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>

#include "nn/MLP.hpp"
#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>

typedef struct _sample {
  std::vector<double> s;	// State
  std::vector<double> pure_a;	// Action computed by the actor
  std::vector<double> a;	// pure_a with gaussian noise for exploration
  std::vector<double> next_s;	// State to which action a leads
  double r;			// Reward
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
    ar& BOOST_SERIALIZATION_NVP(p0);
  }

} sample;

typedef struct _trajectory {
  std::shared_ptr<std::vector<sample>> transitions;  // Transitions
  int cumul_reward;
  int id;
  bool goal_reached;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(transitions);
    ar& BOOST_SERIALIZATION_NVP(cumul_reward);
    ar& BOOST_SERIALIZATION_NVP(id);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

} trajectory;

class AugmentedDENFAC : public arch::AACAgent<MLP, arch::AgentGPUProgOptions> {
  public:
    typedef MLP PolicyImpl;

    AugmentedDENFAC(unsigned int _nb_motors, unsigned int _nb_sensors);

    virtual ~AugmentedDENFAC();

    const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                    bool learning, bool goal_reached, bool last) override;

    void insertSample(const sample& sa);

    /***
     *  Reads meta-paramters from config.ini
     */
    void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override;

    void load_previous_run() override;

    void save_run() override;

    void _start_episode(const std::vector<double>& sensors, bool _learning) override;

    void computePThetaBatch(const std::vector< sample >& vtraj, double *ptheta,const std::vector<double>* all_next_actions);

    void critic_update(uint iter);


    void actor_update_grad();

    void update_actor_critic();

    void end_episode(bool) override;

    void save(const std::string& path, bool) override;

    void load(const std::string& path) override;

    double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override;

    arch::Policy<MLP>* getCopyCurrentPolicy() override;

  protected:

    void _display(std::ostream& out) const override;

    void _dump(std::ostream& out) const override;

  private:
    struct algo_state {
      uint mini_batch_size;
      uint replay_memory;
       
      friend class boost::serialization::access;
      template <typename Archive>
      void serialize(Archive& ar, const unsigned int) {
        ar& BOOST_SERIALIZATION_NVP(mini_batch_size);
        ar& BOOST_SERIALIZATION_NVP(replay_memory);
      }
    };

    // Algorithm parameters
    std::shared_ptr<std::vector<double>> last_action;
    std::shared_ptr<std::vector<double>> last_pure_action;
    std::vector<double> last_state;

    double noise;
    double rmax;
    
    uint episode = 0;
    uint replay_memory, nb_actor_updates, nb_critic_updates, nb_fitted_updates, nb_internal_critic_updates;

    bool gaussian_policy;
    bool learning, reset_qnn, inverting_grad;
    bool reset_ann, keep_weights_wr;

    bool retrace_lambda;

    std::deque<trajectory> trajectories;
    

    // Network parameters

    uint nb_sensors;
    uint replay_traj_size;
    uint batch_norm, weighting_strategy;
    uint last_layer_actor, hidden_layer_type, step;

    std::vector<uint>* hidden_unit_q;
    std::vector<uint>* hidden_unit_a;
    
    double alpha_a;
    double alpha_v;
    double decay_v;
    
    PolicyImpl* ann;
    PolicyImpl* qnn;    
};


#endif