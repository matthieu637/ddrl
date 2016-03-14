#ifndef OFFLINECACLAAG_HPP
#define OFFLINECACLAAG_HPP

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
#include "MLP.hpp"
#include "LinMLP.hpp"
#include "kde.hpp"

#define BATCH_SIZE 40

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  double p0;
  double ptheta_data;
  double pfull_data;

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

class OfflineCaclaAg : public arch::AAgent<> {
 public:
  OfflineCaclaAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : nb_motors(_nb_motors), nb_sensors(_nb_sensors), time_for_ac(1), returned_ac(nb_motors) {

  }

  virtual ~OfflineCaclaAg() {
    delete vnn;
    delete ann;
  }

  const std::vector<double>& run(double r, const std::vector<double>& sensors,
                                bool learning, bool goal_reached) override {

    double reward = r;
    internal_time ++;

    weighted_reward += reward * pow_gamma;
    pow_gamma *= gamma;

    sum_weighted_reward += reward * global_pow_gamma;
    global_pow_gamma *= gamma;

    time_for_ac--;
    if (time_for_ac == 0 || goal_reached) {
      const std::vector<double>& next_action = _run(weighted_reward, sensors, learning, goal_reached);
      time_for_ac = decision_each;

      for (uint i = 0; i < nb_motors; i++)
        returned_ac[i] = next_action[i];

      weighted_reward = 0;
      pow_gamma = 1.f;
    }

    return returned_ac;
  }


  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached) {

    vector<double>* next_action = ann->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning){
      double p0 = 1.f;
      if(gaussian_policy){
        for(uint i=0;i < nb_motors;i++)
          p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise))/(noise * sqrt(2.f * M_PI));
      } else if(did_explore) {
        p0 = noise /* * 0 */;
      } else
        p0 = 1 - noise;
        
      trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0,0});
      proba_s.add_data(last_state);
      trajectory_a.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0,0});
    }

    last_pure_action.reset(new vector<double>(*next_action));
    if(learning) {
      if(gaussian_policy){
        vector<double>* randomized_action = bib::Proba<double>::multidimentionnalGaussianWReject(*next_action, noise);
        delete next_action;
        next_action = randomized_action;
      } else if(bib::Utils::rand01() < noise){ //e-greedy
        for (uint i = 0; i < next_action->size(); i++)
          next_action->at(i) = bib::Utils::randin(-1.f, 1.f);
        did_explore = true;
      } else
        did_explore = false;
    }
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    gamma                   = pt->get<double>("agent.gamma");
    hidden_unit_v           = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a           = pt->get<int>("agent.hidden_unit_a");
    noise                   = pt->get<double>("agent.noise");
    decision_each           = pt->get<int>("agent.decision_each");
    update_pure_ac          = pt->get<bool>("agent.update_pure_ac");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    compare_old_policy      = pt->get<bool>("agent.compare_old_policy");
    update_delta_neg        = pt->get<bool>("agent.update_delta_neg");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    
    if(!gaussian_policy)
      noise = 0.05;

    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
  }

  void start_episode(const std::vector<double>& sensors) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    weighted_reward = 0;
    pow_gamma = 1.d;
    time_for_ac = 1;
    sum_weighted_reward = 0;
    global_pow_gamma = 1.f;
    internal_time = 0;
    
    trajectory_a.clear();
    
    fann_reset_MSE(vnn->getNeuralNet());
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, const OfflineCaclaAg* _ptr) : vtraj(_vtraj), ptr(_ptr) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
      }
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          double nextV = MLP::computeOutVF(local_nn, sm.next_s, {});
          delta += ptr->gamma * nextV;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    const OfflineCaclaAg* ptr;
  };
  
  void computePTheta(vector< sample >& vtraj){
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        sample sm = *it;
        double p0 = 1.f;
        vector<double>* next_action = ann->computeOut(sm.s);
        
        if(gaussian_policy){
          for(uint i=0;i < nb_motors;i++)
            p0 *= exp(-(next_action->at(i)-sm.a[i])*(next_action->at(i)-sm.a[i])/(2.f*noise*noise))/(noise * sqrt(2.f * M_PI));
        } //e greedy policy
        else if( std::equal(next_action->begin(), next_action->end(), sm.a.begin()) )
          p0 = 1 - noise;
        else
          p0 = 0;
          
        it->ptheta_data = p0;
      
        delete next_action;
      }
  }

  void generateSubData(vector< sample >& vtraj_is, const vector< sample >& vtraj_full, uint length){
      std::list<double> weights;
      
      for(auto it = vtraj_full.cbegin(); it != vtraj_full.cend() ; ++it)
          weights.push_back(it->pfull_data);
      
      std::discrete_distribution<int> dist(weights.begin(), weights.end());
      for(uint i = 0; i < length; i++)
        vtraj_is.push_back(vtraj_full[dist(*bib::Seed::random_engine())]);
  }
  
  void update_critic(){
      if (trajectory.size() > 0) {
        //remove trace of old policy

        std::vector<sample> vtraj(trajectory.size());
        std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
        
        //compute 1/u(x)
        
        //compute p_theta(at|xt)
        computePTheta(vtraj);
        
        //compute pfull_data
        for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
            it->pfull_data = (1.f / proba_s.pdf(it->s)) * (it->ptheta_data / it->p0);
//             it->pfull_data = 1.f / (it->ptheta_data) ;
//             it->pfull_data = (1.f / proba_s.pdf(it->s));
//             it->pfull_data = (1.f / proba_s.pdf(it->s)) * it->ptheta_data;
        }
        
        //generate database according to normterm_data
//         std::vector<sample> vtraj_is;
//         generateSubData(vtraj_is, vtraj, BATCH_SIZE);
//         
//         ParraVtoVNext dq(vtraj_is, this);

        auto iter = [&]() {
          //generate database according to normterm_data
          std::vector<sample> vtraj_is;
          generateSubData(vtraj_is, vtraj, BATCH_SIZE);
        
          ParraVtoVNext dq(vtraj_is, this);
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_is.size()), dq);
          
//           LOG_DEBUG("db ok");

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
//           vnn->learn_stoch(dq.data, 10000, 0, 0.0001);
          vnn->learn(dq.data, 10000, 0, 0.0001);
          
          dq.free();
        };

        auto eval = [&]() {
          return fann_get_MSE(vnn->getNeuralNet());
        };

        if(determinist_vnn_update)
              bib::Converger::determinist<>(iter, eval, 30, 0.0001, 0);
        else {
          NN best_nn = nullptr;
          auto save_best = [&]() {
            if(best_nn != nullptr)
              fann_destroy(best_nn);
            best_nn = fann_copy(vnn->getNeuralNet());
          };

          bib::Converger::min_stochastic<>(iter, eval, save_best, 30, 0.0001, 0, 10);
          vnn->copy(best_nn);
          fann_destroy(best_nn);
        }

//         dq.free();
      }
  }

  void end_episode() override {

    MLP* old_vnn;
    if(compare_old_policy){
      old_vnn = new MLP(*vnn);
    }
    
    update_critic();
    
    if (trajectory_a.size() > 0) {

      struct fann_train_data* data = fann_create_train(trajectory_a.size(), nb_sensors, nb_motors);

      uint n=0;
      for(auto it = trajectory_a.begin(); it != trajectory_a.end() ; ++it) {
        sample sm = *it;

        double target = 0.f;
        double mine = 0.f;
        
        if(!compare_old_policy){
          target = sm.r;
          if (!sm.goal_reached) {
            double nextV = vnn->computeOutVF(sm.next_s, {});
            target += gamma * nextV;
          }
          mine = vnn->computeOutVF(sm.s, {});
        } else {
          double newV = vnn->computeOutVF(sm.s, {});
          double oldV = old_vnn->computeOutVF(sm.s, {});
          
          target = newV;
          mine = oldV;
        }

        if(target > mine) {
          for (uint i = 0; i < nb_sensors ; i++)
            data->input[n][i] = sm.s[i];
          if(update_pure_ac){
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.pure_a[i];
          } else {
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.a[i];
          }

          n++;
        } else if(update_delta_neg && !update_pure_ac){
            for (uint i = 0; i < nb_sensors ; i++)
              data->input[n][i] = sm.s[i];
            for (uint i = 0; i < nb_motors; i++)
              data->output[n][i] = sm.pure_a[i];
            n++;
        }
      }

      if(n > 0) {
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, n);

//         ann->learn_stoch(subdata, 5000, 0, 0.0001);
        ann->learn(subdata, 5000, 0, 0.0001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
    }
    
    if(compare_old_policy){
      delete old_vnn;
    }
      
    
  }

  void save(const std::string& path) override {
    ann->save(path+".actor");
    vnn->save(path+".critic");
    bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
    ann->load(path+".actor");
    vnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size() << " " << trajectory_a.size();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }
  

 private:
  uint nb_motors;
  uint nb_sensors;
  uint time_for_ac;

  double weighted_reward;
  double pow_gamma;
  double global_pow_gamma;
  double sum_weighted_reward;
  
  bool update_pure_ac;

  uint internal_time;
  uint decision_each;

  double gamma, noise;
  bool compare_old_policy, gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update, update_delta_neg, did_explore;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::vector<double> returned_ac;

  std::set<sample> trajectory;
  std::set<sample> trajectory_a;
//     std::list<sample> trajectory;
  KDE proba_s;

  MLP* ann;
  MLP* vnn;
};

#endif

