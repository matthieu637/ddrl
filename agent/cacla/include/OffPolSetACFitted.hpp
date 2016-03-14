
#ifndef OFFPOLSETACFITTED_HPP
#define OFFPOLSETACFITTED_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>
#include "MLP.hpp"
#include "LinMLP.hpp"
#include "kde.hpp"

typedef struct _sample {
  std::vector<double> s;
  std::vector<double> pure_a;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
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

class OffPolSetACFitted : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  OffPolSetACFitted(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~OffPolSetACFitted() {
    delete vnn;
    delete ann;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);
    
    if (last_action.get() != nullptr && learning){
      double p0 = 1.f;
      if(!gaussian_policy){
	  if (std::equal(last_action->begin(), last_action->end(), last_pure_action->begin()))//no explo
	    p0 = 1 - noise;
	  else
	    p0 = noise * 0.5f; //should be 0 but p0 > 0
      } else {
	  p0 = 1.f;
	  for(uint i=0;i < nb_motors;i++)
	    p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      }
        
      trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0});
      last_trajectory.insert( {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0});
      proba_s.add_data(last_state);
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
      }
    }
    last_action.reset(next_action);


    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_v           = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a           = pt->get<int>("agent.hidden_unit_a");
    noise                   = pt->get<double>("agent.noise");
    update_pure_ac          = pt->get<bool>("agent.update_pure_ac");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    update_delta_neg        = pt->get<bool>("agent.update_delta_neg");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    change_policy_each      = pt->get<uint>("agent.change_policy_each");
    strategy_w              = pt->get<uint>("agent.strategy_w");
    
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

  void _start_episode(const std::vector<double>& sensors, bool) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;

    internal_time = 0;
    
    //trajectory.clear();
    last_trajectory.clear();
    
    fann_reset_MSE(vnn->getNeuralNet());
    
    if(episode == 0 || episode % change_policy_each == 0){
      if (! boost::filesystem::exists( "polset."+std::to_string(current_loaded_policy) ) ){
        LOG_ERROR("file doesn't exists "<< "polset."+std::to_string(current_loaded_policy));
        exit(1);
      }
      
      ann->load("polset."+std::to_string(current_loaded_policy));
      current_loaded_policy++;
    }
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, const OffPolSetACFitted* _ptr) : vtraj(_vtraj), ptr(_ptr) {
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
    const OffPolSetACFitted* ptr;
  };
  
  void computePTheta(vector< sample >& vtraj, double *ptheta){
    uint i=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      sample sm = *it;
      double p0 = 1.f;
      vector<double>* next_action = ann->computeOut(sm.s);
      
      if(!gaussian_policy){
	  if (std::equal(last_action->begin(), last_action->end(), last_pure_action->begin()))//no explo
	    p0 = 1 - noise;
	  else
	    p0 = noise * 0.5f; //should be 0 but p0 > 0
      } else {
	  p0 = 1.f;
	  for(uint i=0;i < nb_motors;i++)
	    p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      }
	
      ptheta[i] = p0;
      i++;
      delete next_action;
    }
  }
  
  void update_critic(){
      if (trajectory.size() > 0) {
        //remove trace of old policy

        std::vector<sample> *vtraj; 
        double *importance_sample = nullptr; 
        if(strategy_w == 0){
          vtraj = new std::vector<sample>(last_trajectory.size());
          std::copy(last_trajectory.begin(), last_trajectory.end(), vtraj->begin());
          importance_sample = new double [last_trajectory.size()];
          uint i = 0;
          for(auto it = vtraj->begin(); it != vtraj->end() ; ++it) {
            importance_sample[i]=1.000000000000000f;
            i++;
          }
        } else if(strategy_w == 1){
          vtraj = new std::vector<sample>(trajectory.size());
          std::copy(trajectory.begin(), trajectory.end(), vtraj->begin());
          importance_sample = new double [trajectory.size()];
          
          double *ptheta = new double [trajectory.size()];
          //compute p_theta(at|xt)
          computePTheta(*vtraj, ptheta);
          
          uint i=0;
          for(auto it = vtraj->begin(); it != vtraj->end() ; ++it) {
  //             importance_sample[i++] = (1.f / proba_s.pdf(it->s)) * (it->ptheta_data / it->p0);
            importance_sample[i] = ptheta[i];
  //             it->pfull_data = 1.f / (it->ptheta_data) ;
  //             it->pfull_data = (1.f / proba_s.pdf(it->s));
  //             it->pfull_data = (1.f / proba_s.pdf(it->s)) * it->ptheta_data;
            i++;
          }
          delete[] ptheta;
        } else if(strategy_w == 2){
          //compute 1/u(x)
          
        } else {
          LOG_ERROR("fez");
          exit(1);
        }
	
        ParraVtoVNext dq(*vtraj, this);

        auto iter = [&]() {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj->size()), dq);

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
          vnn->learn_stoch_lw(dq.data, importance_sample, 10000, 0, 0.0001);
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

        dq.free(); 
	delete[] importance_sample;
        delete vtraj;
      }
  }

  void end_episode() override {
    update_critic();
    
    episode++;
  }
  
  double criticEval(const std::vector<double>& perceptions) override {
      return vnn->computeOutVF(perceptions, {});
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
        return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise);
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
          8) << std::fixed << std::setprecision(5) << vnn->error() << " " << noise << " " << trajectory.size() ;
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << vnn->error() << " " << trajectory.size() ;
  }
  

 private:
  uint nb_sensors;
  
  bool update_pure_ac;

  uint internal_time;
  uint episode = 0;
  uint current_loaded_policy = 0;
  uint change_policy_each;
  uint strategy_w;

  double noise;
  bool gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update, update_delta_neg;
  uint hidden_unit_v;
  uint hidden_unit_a;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::set<sample> trajectory;
  std::set<sample> last_trajectory;
//     std::list<sample> trajectory;
  KDE proba_s;

  MLP* ann;
  MLP* vnn;
};

#endif

