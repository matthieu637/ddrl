
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
#include "cmaes_interface.h"

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

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);
        
    if (last_action.get() != nullptr && learning){
      double p0 = 1.f;
      for(uint i=0;i < nb_motors;i++)
        p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0};
      trajectory.insert(sa);
      kdtree_s->insert(sa);
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
    
//     if(trajectory.size() % 5 == 0)
//       end_episode();

    return *next_action;
  }


  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map*) override {
    hidden_unit_v           = pt->get<int>("agent.hidden_unit_v");
    hidden_unit_a           = pt->get<int>("agent.hidden_unit_a");
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    lecun_activation        = pt->get<bool>("agent.lecun_activation");
    determinist_vnn_update  = pt->get<bool>("agent.determinist_vnn_update");
    vnn_from_scratch        = pt->get<bool>("agent.vnn_from_scratch");
    converge_precision      = pt->get<double>("agent.converge_precision");
    number_fitted_iteration = pt->get<uint>("agent.number_fitted_iteration");
    
    if(hidden_unit_v == 0)
      vnn = new LinMLP(nb_sensors + nb_motors , 1, 0.0, lecun_activation);
    else
      vnn = new MLP(nb_sensors + nb_motors, hidden_unit_v, nb_sensors, 0.0, lecun_activation);

    fann_set_learning_l2_norm(vnn->getNeuralNet(), 0.05);
    
    if(hidden_unit_a == 0)
      ann = new LinMLP(nb_sensors , nb_motors, 0.0, lecun_activation);
    else
      ann = new MLP(nb_sensors, hidden_unit_a, nb_motors, lecun_activation);
    fann_set_activation_function_output(ann->getNeuralNet(), FANN_LINEAR);
    fann_set_learning_l2_norm(ann->getNeuralNet(), 0.005);
    
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
    ParraVtoVNext(const std::vector<sample>& _vtraj, const NeuralFittedAC* _ptr) : vtraj(_vtraj), ptr(_ptr), actions(_vtraj.size()) {
      data = fann_create_train(vtraj.size(), ptr->nb_sensors+ptr->nb_motors, 1);
      for (uint n = 0; n < vtraj.size(); n++) {
        sample sm = vtraj[n];
        for (uint i = 0; i < ptr->nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        for (uint i= ptr->nb_sensors ; i < ptr->nb_sensors + ptr->nb_motors; i++)
          data->input[n][i] = sm.a[i - ptr->nb_sensors];
        
        actions[n] = ptr->ann->computeOut(sm.next_s);
      }      
    }

    ~ParraVtoVNext() { //must be empty cause of tbb

    }

    void free() {
      fann_destroy_train(data);
      
      for (uint n = 0; n < vtraj.size(); n++)
        delete actions[n];
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {

      struct fann* local_nn = fann_copy(ptr->vnn->getNeuralNet());

      for (size_t n = range.begin(); n < range.end(); n++) {
        sample sm = vtraj[n];

        double delta = sm.r;
        if (!sm.goal_reached) {
          std::vector<double> * next_action = actions[n];
          double nextQA = MLP::computeOutVF(local_nn, sm.next_s, *next_action);
          delta += ptr->gamma * nextQA;
        }

        data->output[n][0] = delta;
      }

      fann_destroy(local_nn);
    }

    struct fann_train_data* data;
    const std::vector<sample>& vtraj;
    const NeuralFittedAC* ptr;
    std::vector<std::vector<double> *> actions;
  };
  
  
  void computePTheta(vector< sample >& vtraj, double *ptheta){
    uint i=0;
    for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
      sample sm = *it;
      vector<double>* next_action = ann->computeOut(sm.s);
      
      double p0 = 1.f;
      for(uint i=0;i < nb_motors;i++)
        p0 *= exp(-(next_action->at(i)-sm.a[i])*(next_action->at(i)-sm.a[i])/(2.f*noise*noise));

      ptheta[i] = p0;
      i++;
      delete next_action;
    }
  }
  
  
  void update_critic(){
      if (trajectory.size() > 0) {
        //remove trace of old policy

        std::vector<sample> vtraj(trajectory.size());
        std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
        //for testing perf - do it before importance sample computation
        std::random_shuffle(vtraj.begin(), vtraj.end());
        
	double *importance_sample = new double [trajectory.size()]; 
        
        double * ptheta = new double [trajectory.size()];
        computePTheta(vtraj, ptheta);
        //compute 1/u(x)
	uint i=0;
        for(auto it = vtraj.begin(); it != vtraj.end() ; ++it) {
	  importance_sample[i] = ptheta[i] / it->p0;
          if(importance_sample[i] > 1.f)
            importance_sample[i] = 1.f;
	  i++;
	}
        
        delete[] ptheta;
        
        //overlearning test
        uint nb_app_sample = (uint) (((float) vtraj.size()) * (100.f/100.f));
        uint nb_test_sample = vtraj.size() - nb_app_sample;
        std::vector<sample> vtraj_app(nb_app_sample);
        std::copy(vtraj.begin(), vtraj.begin() + nb_app_sample, vtraj_app.begin());
        std::vector<sample> vtraj_test(nb_test_sample);
        std::copy(vtraj.begin() + nb_app_sample, vtraj.end(), vtraj_test.begin());
        
        ParraVtoVNext dq(vtraj_app, this);
        ParraVtoVNext dq_test(vtraj_test, this);

        auto iter = [&]() {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_app.size()), dq);
//           dq(tbb::blocked_range<size_t>(0, vtraj.size()));

          if(vnn_from_scratch)
            fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
          
//             vnn->learn_stoch_lw(dq.data, importance_sample, 10000, 0, converge_precision);
          vnn->learn_stoch(dq.data, 100, 0, converge_precision);
        };

        auto eval = [&]() {
          double app_error = fann_get_MSE(vnn->getNeuralNet());
          tbb::parallel_for(tbb::blocked_range<size_t>(0, vtraj_test.size()), dq_test);
          
          double* out;
          double test_error = 0.f;
          for(uint j = 0; j < dq_test.data->num_data; j++){
            out = fann_run(vnn->getNeuralNet(), dq_test.data->input[j]);
            
            double l = (dq_test.data->output[j][0] - out[0]);
//             test_error += l*l;
            
            test_error += l*l * importance_sample[j + nb_app_sample];
          }
          
          
//           LOG_DEBUG(app_error << " " << test_error);
          
          if(dq_test.data->num_data == 0)
            return app_error;
          
          test_error = sqrt(test_error) / vtraj_test.size();
          return test_error;
//           return app_error;
        };

        if(determinist_vnn_update){
//           iter();
          bib::Converger::determinist<>(iter, eval, number_fitted_iteration, converge_precision, 3, "deter_critic");
        } else {
          NN best_nn = nullptr;
          auto save_best = [&]() {
            if(best_nn != nullptr)
              fann_destroy(best_nn);
            best_nn = fann_copy(vnn->getNeuralNet());
          };

          bib::Converger::min_stochastic<>(iter, eval, save_best, 30, converge_precision, 1, 10, "stoch_crtic");
          vnn->copy(best_nn);
          fann_destroy(best_nn);
        }

        dq.free(); 
        dq_test.free();
	delete[] importance_sample;
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
  
  void update_actorN2(){
    std::vector<sample> vtraj(trajectory.size());
    std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
    
    struct fann_train_data* data = fann_create_train(vtraj.size(), nb_sensors, nb_motors);
    uint database_size = 0;
    
    for(auto it_s = vtraj.begin(); it_s != vtraj.end() ; ++it_s){
      double max = -500000000000000000.f;
      std::vector<double>* best_a = &(vtraj[0].a);
      
      for (uint i = 0; i < nb_sensors ; i++)
        data->input[database_size][i] = it_s->s[i];
  
      for(auto it_a = vtraj.begin(); it_a != vtraj.end() ; ++it_a){
          double delta = vnn->computeOutVF(it_s->s, it_a->a);
          if(max <= delta){
              max = delta;
              best_a = &it_a->a;
          }
      }
      
      for (uint i = 0; i < nb_motors; i++)
        if(best_a->at(i) >= 1.f)
          data->output[database_size][i] = 1.f;
        else if(best_a->at(i) <= -1.f)
          data->output[database_size][i] = -1.f;
        else
          data->output[database_size][i] = best_a->at(i);
      
      database_size++;
    }
    
    ann->learn_stoch(data, 10000, 0, 0.00001);

    fann_destroy_train(data);
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
//           double current_max = 
          Mx(*delta, it->s, sigma, imax);
          
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
//             current_max = 
            Mx(delta_candidate, it->s, sigma, imax);
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
//       bib::Logger::PRINT_ELEMENTS(sigma);
      
      
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
      std::vector<sample> vtraj(trajectory.size());
      std::copy(trajectory.begin(), trajectory.end(), vtraj.begin());
      std::random_shuffle(vtraj.begin(), vtraj.end());
      
      uint nb_app_sample = (uint) (((float) trajectory.size()) * (100.f/100.f));
      struct fann_train_data* data = fann_create_train(nb_app_sample, nb_sensors, nb_motors);
      
      uint n=0;
      for(uint i = 0; i < nb_app_sample; i++) {
        sample sm = vtraj[i];
        
        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = 0.f;//sm.a[i]; //don't care

        n++;
      }
      
      datann_derivative d = {vnn, (int)nb_sensors, (int)nb_motors};
      
      auto iter = [&]() {
//        fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn, &d);
#warning inverting_gradient      
        fann_train_epoch_irpropm_gradient(ann->getNeuralNet(), data, derivative_nn_inverting, &d);
      };

      auto eval = [&]() {
        //compute weight sum
//         return ann->weight_l1_norm();
        return fitfun_sum_overtraj();
      };

      bib::Converger::determinist<>(iter, eval, 10, 0.0001, 10, "actor grad");
      
      //alone update
//       iter();
      
      fann_destroy_train(data);
      
    }
  }
  
  void update_actor_discretization(uint){
      struct fann_train_data* data = fann_create_train(trajectory.size(), nb_sensors, nb_motors);
      
      uint n=0;
      for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
        sample sm = *it;
        
        for (uint i = 0; i < nb_sensors ; i++)
          data->input[n][i] = sm.s[i];
        
        std::vector<double>* na = vnn->optimizedBruteForce(sm.s,0.01);
        
        for (uint i = 0; i < nb_motors; i++)
          data->output[n][i] = na->at(i);
        
//         LOG_FILE("debug_discret." + std::to_string(episode)+ "."+std::to_string(minep), sm.s[0] << " "<< na->at(0));
        
        delete na;

        n++;
      }
      
      ann->learn_stoch(data, 10000, 0, 0.00000001);

      fann_destroy_train(data);
  }
  
  bool is_feasible(const double* parameters, uint N){
      for(uint i=0;i < N;i++)
        if(fabs(parameters[i]) >= 5.f){
          return false;
        }
        
      return true;
  }
  
  double fitfun_sum_overtraj_debug(){
    double sum = 0;
    for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
      sample sm = *it;
      
      vector<double>* next_action = ann->computeOut(sm.s);
      
      sum += vnn->computeOutVF(sm.s, *next_action);
      delete next_action;
    }
    
    return sum / trajectory.size();
  }
  
  double fitfun_sum_overtraj(){
    double sum = 0;
    for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
      sample sm = *it;
      
      vector<double>* next_action = ann->computeOut(sm.s);
      
      sum += vnn->computeOutVF(sm.s, *next_action);
//       sum += vnn->computeOutVF(sm.s, *next_action) * (1.f/proba_s.pdf(sm.s));
      
      delete next_action;
    }
    
    return sum / trajectory.size();
  }
  
  double fitfun(double const *x, int N) {
      const uint dimension = N;
      
      struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
      fann_get_connection_array(ann->getNeuralNet(), connections);
      
      ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
      for(uint j=0; j< dimension; j++)
	connections[j].weight=x[j];
      
      fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
      free(connections);
      
      double sum = fitfun_sum_overtraj();
      
      //LOG_DEBUG(sum);
      return -sum;
  }
  
  void update_actor_cmaes(){
      cmaes_t evo;
      double *arFunvals, *const*pop;
      const double* xfinal;
      uint i; 
      
      struct fann_connection* connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
      fann_get_connection_array(ann->getNeuralNet(), connections);
      const uint dimension = (nb_sensors+1)*hidden_unit_a + (hidden_unit_a+1)*nb_motors;
      double* startx  = new double[dimension];
      double* deviation  = new double[dimension];
      for(uint j=0; j< dimension; j++){
          startx[j] = connections[j].weight;
          deviation[j] = 0.3;
      }
      free(connections);
      
      uint population = 15;
      uint generation = 50;
      arFunvals = cmaes_init(&evo, dimension, startx, deviation, 0, population, NULL);
      
      //run cmaes
      while(!cmaes_TestForTermination(&evo) && generation > 0 )
      {
        pop = cmaes_SamplePopulation(&evo);
        generation--;
          
        for (i = 0; i < population; ++i)
//           while (!is_feasible(pop[i], dimension)){
            cmaes_ReSampleSingle(&evo, i);
//         }
        
        for (i = 0; i < cmaes_Get(&evo, "lambda"); ++i) {
    	  arFunvals[i] = fitfun(pop[i], dimension);
        }
        
        cmaes_UpdateDistribution(&evo, arFunvals);
        
        if(cmaes_TestForTermination(&evo))
          LOG_INFO("mismatch "<< cmaes_TestForTermination(&evo));
      }
      
      //get final solution
      xfinal = cmaes_GetPtr(&evo, "xbestever");
      
      connections = (struct fann_connection*) calloc(fann_get_total_connections(ann->getNeuralNet()), sizeof(struct fann_connection));
      fann_get_connection_array(ann->getNeuralNet(), connections);
      
      ASSERT(dimension == fann_get_total_connections(ann->getNeuralNet()), "dimension mismatch");
      for(uint j=0; j< dimension; j++)
	connections[j].weight=xfinal[j];
      
      fann_set_weight_array(ann->getNeuralNet(), connections, dimension);
      free(connections);
  }
  
  uint episode=0;
  void end_episode() override {
    
    if(!learning)
      return;
    
    episode++;

    int* resetlooping;
    auto iter = [&]() {
        LOG_DEBUG(ann->weight_l1_norm() << " " << fitfun_sum_overtraj() << " " << fann_get_MSE(vnn->getNeuralNet()) <<" "  << vnn->weight_l1_norm());
        update_critic();

        update_actor_nfqca();
//           update_actor_cmaes();
//           update_actor_discretization(*resetlooping);
//           update_actorN2();
        LOG_FILE("test" + std::to_string(episode) , fitfun_sum_overtraj() << " " << fann_get_MSE(vnn->getNeuralNet()));
        
        if(ann->weight_l1_norm() > 250.f && false){
          fann_randomize_weights(ann->getNeuralNet(), -0.025, 0.025);
          fann_randomize_weights(vnn->getNeuralNet(), -0.025, 0.025);
          LOG_DEBUG("actor failed");
          *resetlooping = 0;
        }
    };
    
//       auto eval = [&]() {
//         
//         
//         double sr = fitfun_sum_overtraj();
// //         if(fann_get_MSE(vnn->getNeuralNet()) < 0.001)
//           return -sr;
// //         else
// //           return -sr+100. ;
// //         return fann_get_MSE(vnn->getNeuralNet());
//       };
//       
//       NN best_ann = nullptr;
//       NN best_vnn = nullptr;
//       auto save_best = [&]() {
//         if(best_ann != nullptr){
//           fann_destroy(best_ann);
//           fann_destroy(best_vnn);
//         }
//         best_ann = fann_copy(ann->getNeuralNet());
//         best_vnn = fann_copy(vnn->getNeuralNet());
//       };
//       
// //       bib::Converger::min_stochastic_with_neg<>(iter, eval, save_best, 200, converge_precision, 0, 80, "ac_inter");
//       vnn->copy(best_vnn);
//       ann->copy(best_ann);
//       fann_destroy(best_ann);
//       fann_destroy(best_vnn);
    
    int k = 0;
    resetlooping = &k;
    for(k =0;k<10;k++)
      iter();
    
    LOG_DEBUG(ann->weight_l1_norm());
    
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
//     ann->load(path+".actor");
//     vnn->load(path+".critic");
    ann->load(path);
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

  double noise, converge_precision;
  bool gaussian_policy, vnn_from_scratch, lecun_activation, 
        determinist_vnn_update;
  uint hidden_unit_v;
  uint hidden_unit_a;
  uint number_fitted_iteration;
  
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
  
  struct L2_distance
  {
    typedef double distance_type;
    
    double operator() (const double& __a, const double& __b, const size_t) const
    {
      double d = (__a - __b);
      d = sqrt(d*d);
      return d;
    }
  };
  
  typedef KDTree::KDTree<sample, KDTree::_Bracket_accessor<sample>, L2_distance> kdtree_sample;
  kdtree_sample* kdtree_s;

  MLP* ann;
  MLP* vnn;
};

#endif

