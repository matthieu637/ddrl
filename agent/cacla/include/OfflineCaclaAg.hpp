#ifndef OFFLINECACLAAG_HPP
#define OFFLINECACLAAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "nn/MLP.hpp"

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;

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

  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }

    for (uint i = 0; i < a.size(); i++) {
      if(a[i] != b.a[i])
        return a[i] < b.a[i];
    }

    return false;
  }

} sample;

class OfflineCaclaAg : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;

  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors), empty_action(0) {

  }

  virtual ~OfflineCaclaAg() {
    delete vnn;
    delete ann;

    delete hidden_unit_v;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                  bool learning, bool goal_reached, bool last) {

    vector<double>* next_action = ann->computeOut(sensors);

    if (last_action.get() != nullptr && learning)
      trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached || last});

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy) {
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalTruncatedGaussian(*next_action, noise);
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
    hidden_unit_v           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_v"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    update_pure_ac          = pt->get<bool>("agent.update_pure_ac");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    update_delta_neg        = pt->get<bool>("agent.update_delta_neg");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    update_critic_first     = pt->get<bool>("agent.update_critic_first");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");
    stoch_iter              = pt->get<uint>("agent.stoch_iter");
    batch_norm              = pt->get<uint>("agent.batch_norm");
    actor_output_layer_type = pt->get<uint>("agent.actor_output_layer_type");
    hidden_layer_type       = pt->get<uint>("agent.hidden_layer_type");
    alpha_a                 = pt->get<double>("agent.alpha_a");
    alpha_v                 = pt->get<double>("agent.alpha_v");

    vnn = new MLP(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, 1, -1, hidden_layer_type, batch_norm);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, alpha_a, 1, hidden_layer_type, actor_output_layer_type, batch_norm, true);
  }

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    trajectory.clear();
  }

  void update_critic() {
    if (trajectory.size() > 0) {
      //remove trace of old policy
      auto iter = [&]() {
        std::vector<double> all_states(trajectory.size() * nb_sensors);
        std::vector<double> all_next_states(trajectory.size() * nb_sensors);
        std::vector<double> v_target(trajectory.size());
        uint li=0;
        for (auto it : trajectory) {
          std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
          std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
          li++;
        }

        auto all_nextV = vnn->computeOutVFBatch(all_next_states, empty_action);

        li=0;
        for (auto it : trajectory) {
          double delta = it.r;
          if (!it.goal_reached) {
            double nextV = all_nextV->at(li);
            delta += gamma * nextV;
          }

          v_target[li] = delta;
          li++;
        }

        ASSERT(li == trajectory.size(), "");
        if(vnn_from_scratch){
          delete vnn;
          vnn = new MLP(nb_sensors, nb_sensors, *hidden_unit_v, alpha_v, trajectory.size(), -1, hidden_layer_type, batch_norm);
        }
        vnn->learn_batch(all_states, empty_action, v_target, stoch_iter);
        
        delete all_nextV;
      };

      for(uint i=0; i<number_fitted_iteration; i++)
        iter();
    }
  }

  void end_episode() override {
//     LOG_FILE("policy_exploration", ann->hash());

    if(trajectory.size() > 0)
      vnn->increase_batchsize(trajectory.size());
    
    if(update_critic_first)
      update_critic();

    if (trajectory.size() > 0) {
      std::vector<double> sensors(trajectory.size() * nb_sensors);
      std::vector<double> actions(trajectory.size() * nb_motors);

      std::vector<double> all_states(trajectory.size() * nb_sensors);
      std::vector<double> all_next_states(trajectory.size() * nb_sensors);
      uint li=0;
      for (auto it : trajectory) {
        std::copy(it.s.begin(), it.s.end(), all_states.begin() + li * nb_sensors);
        std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + li * nb_sensors);
        li++;
      }

      auto all_nextV = vnn->computeOutVFBatch(all_next_states, empty_action);
      auto all_mine = vnn->computeOutVFBatch(all_states, empty_action);

      uint n=0;
      li=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;

        target = sm.r;
        if (!sm.goal_reached) {
//           double nextV = vnn->computeOutVF(sm.next_s, {});
          double nextV = all_nextV->at(li);
          target += gamma * nextV;
        }
//         mine = vnn->computeOutVF(sm.s, {});
        mine = all_mine->at(li);

        if(target > mine) {
          std::copy(it->s.begin(), it->s.end(), sensors.begin() + n * nb_sensors);

          if(update_pure_ac)
            std::copy(it->pure_a.begin(), it->pure_a.end(), actions.begin() + n * nb_motors);
          else
            std::copy(it->a.begin(), it->a.end(), actions.begin() + n * nb_motors);

          n++;
        } else if(update_delta_neg && !update_pure_ac) {
          std::copy(it->s.begin(), it->s.end(), sensors.begin() + n * nb_sensors);
          std::copy(it->pure_a.begin(), it->pure_a.end(), actions.begin() + n * nb_motors);
          n++;
        }
        li++;
      }

      if(n > 0) {
        ann->increase_batchsize(n);
        ann->learn_batch(sensors, empty_action, actions, stoch_iter);
      }

      delete all_nextV;
      delete all_mine;
    }

    if(!update_critic_first){
      update_critic();
    }
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
    bib::XMLEngine::save<>(trajectory, "trajectory", path+".trajectory");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
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
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }

 private:
  uint nb_sensors;

  bool update_pure_ac;

  double noise;
  bool gaussian_policy, vnn_from_scratch, update_critic_first,
       update_delta_neg;
  uint number_fitted_iteration, stoch_iter;
  uint batch_norm, actor_output_layer_type, hidden_layer_type;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;
  double alpha_v, alpha_a;

  std::set<sample> trajectory;
//     std::list<sample> trajectory;

  MLP* ann;
  MLP* vnn;

  std::vector<uint>* hidden_unit_v;
  std::vector<uint>* hidden_unit_a;
  std::vector<double> empty_action;
};

#endif

