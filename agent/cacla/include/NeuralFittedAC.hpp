
#ifndef NEURALFITTEDAC_HPP
#define NEURALFITTEDAC_HPP

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
#include "kdtree++/kdtree.hpp"

#define DOUBLE_COMPARE_PRECISION 1e-9

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

  //Used to store all sample into a tree, might be stochastic
  //only pure_a is negligate
  bool operator< (const _sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return s[i] < b.s[i];
    }
    
    for (uint i = 0; i < a.size(); i++) {
      if(fabs(a[i] - b.a[i])>=DOUBLE_COMPARE_PRECISION)
        return a[i] < b.a[i];
    }
    
    for (uint i = 0; i < next_s.size(); i++) {
      if(fabs(next_s[i] - b.next_s[i])>=DOUBLE_COMPARE_PRECISION)
        return next_s[i] < b.next_s[i];
    }
    
    if(fabs(r - b.r)>=DOUBLE_COMPARE_PRECISION)
        return r < b.r;

    return goal_reached < b.goal_reached;
  }
  
  typedef double value_type;

  inline double operator[](size_t const N) const{
        return s[N];
  }
  
  bool same_state(const _sample& b) const{
    for (uint i = 0; i < s.size(); i++) {
      if(fabs(s[i] - b.s[i])>=DOUBLE_COMPARE_PRECISION)
        return false;
    }
    
    return true;
  }

} sample;

class NeuralFittedAC : public arch::AACAgent<MLP, arch::AgentProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  NeuralFittedAC(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, arch::AgentProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~NeuralFittedAC() {
    delete kdtree_s;
    
    delete vnn;
    delete ann;
  }
  
  vector<double>* policy(const std::vector<double>& sensors) const{
     vector<double>* next_action = ann->computeOut(sensors);
    
//     if(trajectory.size() < 4){
//       return ann->computeOut(sensors);
//     }
//     
//     
//     //tant que plus proche et augmente => candidats
//     
//     uint n=0;
//     for(auto it : vtraj){
//       if(sa.same_state(it) && delta[n] > mydelta)
//         return false;
//       n++;
//     }
//     
//     
//     std::vector<sample> vtraj(trajectory.size());
//     std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
//       
//     //first update sigma
//     double min_delta, max_delta;
//     std::vector<double>* delta = computeDeltaCurrentTraj(vtraj, min_delta, max_delta);
//     
//     for (uint i = 0; i < nb_sensors ; i++){
//       sigma[i] = 0.000000001f;
//     }
//     uint imax=0;
//     double current_max = Mx(*delta, sensors, sigma, imax);
//     
//     vector<double>* next_action = new std::vector<double>(vtraj[imax].a);
//     
//         struct UniqTest
//     {
//       const std::vector<double>& _s;
//       OffVSetACFitted* ptr;
//       double _min_delta;
// 
//       bool operator()(const sample& t ) const
//       {
//           if( !( t != _sa))
//             return false;
//           
//           double delta_ns= t.r;
//           if(!t.goal_reached){
//             double nextV = ptr->vnn->computeOutVF(t.next_s, {});
//             delta_ns += ptr->gamma * nextV;
//           }
//           
//           if(delta_ns <= _min_delta)
//             return false;
//           
//           return true;
//       }
//     };
//     UniqTest t={sensors, this, min_delta};
//     auto closest = kdtree_s->find_nearest_if(sensors, std::numeric_limits<double>::max(), t);
//     
     
     return next_action;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = policy(sensors);

    if (last_action.get() != nullptr && learning){
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached};
      trajectory.insert(sa);
      kdtree_s->insert(sa);
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
    
    if(trajectory.size() % 5 == 0)
      end_episode();

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    gamma                   = pt->get<double>("agent.gamma");
    hidden_unit_v           = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a           = pt->get<int>("agent.hidden_unit_a");
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    
    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
    
    sigma.resize(nb_sensors);
    for (uint i = 0; i < nb_sensors ; i++){
      sigma[i] = bib::Utils::rand01();
    }
    
    kdtree_s = new kdtree_sample(nb_sensors);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    //trajectory.clear();
    
    fann_reset_MSE(vnn->getNeuralNet());
    learning = _learning;
  }

  struct ParraVtoVNext {
    ParraVtoVNext(const std::vector<sample>& _vtraj, const NeuralFittedAC* _ptr) : vtraj(_vtraj), ptr(_ptr) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors+ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];
      }
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());
      struct fann* local_pol = fann_copy(ptr->ann->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          std::vector<double> * next_action = MLP::computeOut(local_pol, sm.next_s);
//           std::vector<double> * next_action = ptr->policy(sm.next_s);
          double nextQA = MLP::computeOutVF(local_nn, sm.next_s, *next_action);
          delta += ptr->gamma * nextQA;
          delete next_action;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
      fann_destroy(local_pol);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    const NeuralFittedAC* ptr;
  };
  
  void update_critic(){
      if (trajectory.size() > 0) {
        //remove trace of old policy

        std::vector<sample> vtraj(trajectory.size());
        std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
// 	double *importance_sample = new double [trajectory.size()]; 
        
        //compute 1/u(x)
        //TODO:normalize data?
// 	uint i=0;
//         for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
// 	  importance_sample[i] = 1.00000000000f;
// 	  i++;
// 	}
        
        ParraVtoVNext dq(vtraj, this);

        auto iter = [&]() {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj.size()), dq);
//           dq(tbb::blocked_range<size_t>(0, vtraj.size()));

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
//           vnn->learn_stoch_lw(dq.data, importance_sample, 10000, 0, 0.0001);
          vnn->learn_stoch(dq.data, 10000, 0, 0.00001);
        };

        auto eval = [&]() {
          return fann_get_MSE(vnn->getNeuralNet());
        };

        if(determinist_vnn_update)
              bib::Converger::determinist<>(iter, eval, 30, 0.00001, 0, "deter_critic");
        else {
          NN best_nn = nullptr;
          auto save_best = [&]() {
            if(best_nn != nullptr)
              fann_destroy(best_nn);
            best_nn = fann_copy(vnn->getNeuralNet());
          };

          bib::Converger::min_stochastic<>(iter, eval, save_best, 30, 0.0001, 0, 10, "stoch_crtic");
          vnn->copy(best_nn);
          fann_destroy(best_nn);
        }

        dq.free(); 
// 	delete[] importance_sample;
//         LOG_DEBUG("critic updated");
      }
  }
  
  
  double Mx(const std::vector<double>& delta, const std::vector<double>& x, const std::vector<double>& sigma, uint& imax) const{
    double max_gauss = std::numeric_limits<double>::lowest();
    uint i=0;
    for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
      const std::vector<double>& s = it->s;
      double v = delta[i];
      double toexp = 0;
      for(uint j=0;j< x.size(); j++){
        ASSERT(sigma[j] > 0.000000000f, "wrong sigma " << sigma[j]);
        toexp += ((s[j] - x[j])*(s[j] - x[j]))/(sigma[j]*sigma[j]);
      }
      toexp *= -0.5f;
      v = v * exp(toexp);
      
      if(v>=max_gauss){
        max_gauss = v;
        imax=i;
      }
      i++;
    }
    
    return max_gauss;
  }
  
  std::vector<double>* computeDeltaCurrentTraj(const std::vector<sample>& vtraj, double& min_delta, double& max_delta) const{
    std::vector<double> *delta = new std::vector<double>(vtraj.size());
    min_delta = std::numeric_limits<double>::max();
    max_delta = std::numeric_limits<double>::lowest();
    
    uint n=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      sample sm = *it;
      delta->at(n) = vnn->computeOutVF(sm.s, sm.a);
      
      if(delta->at(n) <= min_delta)
        min_delta = delta->at(n);
      if(delta->at(n) >= max_delta)
        max_delta = delta->at(n);
      
      n++;
    }
    
    n=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      delta->at(n) = bib::Utils::transform(delta->at(n), min_delta, max_delta, 0, 1);
      n++;
    }
    
    return delta;
  }
  
  bool isTheBestInCurrentState(const sample& sa, const std::vector<sample>& vtraj, const std::vector<double>& delta, double mydelta) const{
//     even with a kd-tree, it is requiered to either compute again the score or iterate over all data
    
//     auto closest = kdtree_s->find_nearest(sa);
//     if(!sa.same_state(*closest.first)) //the nearest is not in the same state, so i'm the best
//       return true;
    
    //nearest in the same state, we must compare our score
    uint n=0;
    for(auto it : vtraj){
      if(sa.same_state(it) && delta[n] > mydelta)
        return false;
      n++;
    }
    
    return true;
  }
  
  
  
  void update_actor(){
   
    if (trajectory.size() > 4) {
      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
        
      //first update sigma
      double min_delta, max_delta;
      std::vector<double>* delta = computeDeltaCurrentTraj(vtraj, min_delta, max_delta);
    
      for(uint force=0;force < 20; force++){
      uint n=0;
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        //don't care about me if i'm the worst overall
        //or if i'm not the best in the current state
        if(delta->at(n) > DOUBLE_COMPARE_PRECISION && isTheBestInCurrentState(*it, vtraj, *delta, delta->at(n))){
          uint imax=0;
          double current_max = Mx(*delta, it->s, sigma, imax);
          
          double current_efficacity = bib::Utils::transform(vnn->computeOutVF(it->s, vtraj[imax].a), min_delta, max_delta, 0., 1.);
          double own_efficacity = delta->at(n);
        
          //i am better than the max in my position but it's not me
          if(imax != n && own_efficacity > current_efficacity){//so sigma should be decreased
            double ln = own_efficacity > current_efficacity ? log(own_efficacity / current_efficacity) : log(current_efficacity / own_efficacity);
            ln = 2.f * ln;
            ln = ln * 4.f;
                
            for (uint i = 0; i < nb_sensors ; i++){
                double sq = (it->s[i] - vtraj[imax].s[i])*(it->s[i] - vtraj[imax].s[i]);
                double wanted_sigma = sqrt(sq/ln);
                  
//                 sigma[i] = sigma[i] - 0.001*fabs(it->s[i] - vtraj[imax].s[i]);
                sigma[i] = sigma[i] + 0.0001*(wanted_sigma - sigma[i]);
            }
          } else if(imax == n) {
            //i am the max, so i should check that the max without me is indeed worst than me
            
            std::vector<double> delta_candidate(delta->size());
            std::copy(delta->begin(), delta->end(), delta_candidate.begin());
            for(uint i=0;i<delta_candidate.size(); i++){
              if(it->same_state(vtraj[i]) || delta->at(i) <= delta->at(n) )
                delta_candidate[i]=0.f;
            }
            
            imax=delta_candidate.size();
            current_max = Mx(delta_candidate, it->s, sigma, imax);
            if(imax < delta_candidate.size()){ //check candidate exists
              
              double efficacity_without_me = bib::Utils::transform(vnn->computeOutVF(it->s, vtraj[imax].a), min_delta, max_delta, 0., 1.);
              
              //i'm the max, but the max without me is better than me
              if(own_efficacity < efficacity_without_me){ // so sigma should be increase to drown me
                
                double ln = own_efficacity > efficacity_without_me ? log(own_efficacity / efficacity_without_me) : log(efficacity_without_me / own_efficacity);
                ln = 2.f * ln;
                ln = ln * 4.f;
                for (uint i = 0; i < nb_sensors ; i++){
                  double sq = (it->s[i] - vtraj[imax].s[i])*(it->s[i] - vtraj[imax].s[i]);
                  double wanted_sigma = sqrt(sq/ln);
                  
                  sigma[i] = sigma[i] + 0.0001*(wanted_sigma - sigma[i]);
//                   sigma[i] = sigma[i] + 0.001*fabs(it->s[i] - vtraj[imax].s[i]);
                }
              }
            }
          }
        }
        n++;
      }
      
      }
      bib::Logger::PRINT_ELEMENTS(sigma);
      
      
      //now that sigma has been updated
      //compute importance of each action for the policy
      struct fann_train_data* data = fann_create_train(vtraj.size(), nb_sensors, nb_motors);
      
//       std::ofstream out;
//       out.open("debugMxP", std::ofstream::out | std::ofstream::trunc);
      
      uint n=0;
      uint database_size=0;
      for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
        sample sm = *it;
        
        uint imax=0;
        Mx(*delta, it->s, sigma, imax);
        
//         out << it->s[0] << " " << delta->at(n) << " " << it->a[0] << " ";
//         if(imax == n)
//           out << "1" ;
//         else out << "0";
//         out << std::endl;
        
        if(imax == n){//if i'm not the max, don't learn me

          for (uint i = 0; i < nb_sensors ; i++)
            data->input[database_size][i] = sm.s[i];
          
          for (uint i = 0; i < nb_motors; i++)
            data->output[database_size][i] = sm.a[i];

          database_size++;
        }
        n++;
      }
      
      if(database_size > 0){
        struct fann_train_data* subdata = fann_subset_train_data(data, 0, database_size);

        ann->learn_stoch(subdata, 10000, 0, 0.00001);

        fann_destroy_train(subdata);
      }
      fann_destroy_train(data);
      
      delete delta;
      
    }
    
  }

  void update_actor_nfqca(){
    if(trajectory.size() > 0) {
      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);
      
      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;
        
        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = 0.f;//sm.a[i]; //don't care

        n++;
      }
      
      datann_derivative d = {vnn, (int)nb_sensors, (int)nb_motors};
    
//       auto iter = [&]() {
//         fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
//       };
// 
//       auto eval = [&]() {
//         //compute weight sum
//         return ann->weight_l1_norm();
//       };
// 
//       bib::Converger::determinist<>(iter, eval, 1, 0.0001, 25, "actor");
      fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
      
      fann_destroy_train(data);
    }
  }
  
  void end_episode() override {
    
    if(!learning)
      return;
    
    update_critic();
    
//     LOG_DEBUG("critic updated");
    
//     update_actor();
//     if(trajectory.size()%10==0)
    update_actor_nfqca();
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return vnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
        return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
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

  double gamma, noise;
  bool gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update;
  uint hidden_unit_v;
  uint hidden_unit_a;
  
  bool learning;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::vector<double> sigma;

  std::set<sample> trajectory;
//     std::list<sample> trajectory;
  KDE proba_s;
  
  struct L1_distance
  {
    typedef double distance_type;
    
    double operator() (const double& __a, const double& __b, const size_t) const
    {
      double d = fabs(__a - __b);
      return d;
    }
  };
  typedef KDTree::KDTree<sample, KDTree::_Bracket_accessor<sample>, L1_distance> kdtree_sample;
  kdtree_sample* kdtree_s;

  MLP* ann;
  MLP* vnn;
};

#endif

