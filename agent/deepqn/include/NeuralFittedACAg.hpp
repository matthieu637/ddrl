


#ifndef NEURALFITTEDACAG_HPP
#define NEURALFITTEDACAG_HPP

#include <vector>
#include <string>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/filesystem.hpp>

#include "MLP.hpp"
#include "arch/AACAgent.hpp"
#include "bib/Seed.hpp"
#include "bib/Utils.hpp"
#include <bib/MetropolisHasting.hpp>
#include <bib/XMLEngine.hpp>


class NeuralFittedACAg : public arch::AACAgent<MLP, AgentGPUProgOptions> {
 public:
  typedef MLP PolicyImpl;
   
  NeuralFittedACAg(unsigned int _nb_motors, unsigned int _nb_sensors)
    : arch::AACAgent<MLP, AgentGPUProgOptions>(_nb_motors), nb_sensors(_nb_sensors) {

  }

  virtual ~NeuralFittedACAg() {
    delete qnn;
    delete ann;
    
    delete hidden_unit_q;
    delete hidden_unit_a;
  }

  const std::vector<double>& _run(double reward, const std::vector<double>& sensors,
                                 bool learning, bool goal_reached, bool) {

    vector<double>* next_action = ann->computeOut(sensors);

#warning inverting
      for(uint i=0;i < nb_motors ; i++)
        if(next_action->at(i) > 1.f)
          next_action->at(i)=1.f;
        else if(next_action->at(i) < -1.f)
          next_action->at(i)=-1.f;
    
    if (last_action.get() != nullptr && learning){
      double p0 = 1.f;
      for(uint i=0;i < nb_motors;i++)
        p0 *= exp(-(last_pure_action->at(i)-last_action->at(i))*(last_pure_action->at(i)-last_action->at(i))/(2.f*noise*noise));
      
      sample sa = {last_state, *last_pure_action, *last_action, sensors, reward, goal_reached, p0, 0.};
      insertSample(sa);

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

  void insertSample(const sample& sa){
    
    if(pure_online){
      if(trajectory.size() >= replay_memory)
        trajectory.pop_front();
      trajectory.push_back(sa);
      
//       end_episode();//not here
    } else {
      last_trajectory.push_back(sa);
    }
  }

  void _unique_invoke(boost::property_tree::ptree* pt, boost::program_options::variables_map* command_args) override {
    hidden_unit_q           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_q"));
    hidden_unit_a           = bib::to_array<uint>(pt->get<std::string>("agent.hidden_unit_a"));
    noise                   = pt->get<double>("agent.noise");
    gaussian_policy         = pt->get<bool>("agent.gaussian_policy");
    kMinibatchSize          = pt->get<uint>("agent.mini_batch_size");
    pure_online             = true;
    replay_memory           = 50000;
    
    if(command_args->count("gpu") == 0 || command_args->count("cpu") > 0){
      caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
      LOG_INFO("CPU mode");
    } else {
      caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
      caffe::Caffe::SetDevice(0);
      LOG_INFO("GPU mode");
    }
    
    qnn = new MLP(nb_sensors + nb_motors, nb_sensors, *hidden_unit_q, 0.0001, kMinibatchSize);

    ann = new MLP(nb_sensors, *hidden_unit_a, nb_motors, 0.0001, kMinibatchSize);
  }

  void _start_episode(const std::vector<double>& sensors, bool _learning) override {
    last_state.clear();
    for (uint i = 0; i < sensors.size(); i++)
      last_state.push_back(sensors[i]);

    last_action = nullptr;
    last_pure_action = nullptr;
    
    learning = _learning;
  }
  
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

  double fitfun_sum_overtraj(){
    double sum = 0;
    for(auto it = trajectory.begin(); it != trajectory.end() ; ++it) {
      sample sm = *it;
      
      vector<double>* next_action = ann->computeOut(sm.s);
      
      sum += qnn->computeOutVF(sm.s, *next_action);
      delete next_action;
    }
    
    return sum / trajectory.size();
  }
  
  void sample_transition(std::vector<sample>& traj){
     for(uint i=0;i<traj.size();i++){
       int r = std::uniform_int_distribution<int>(0, traj.size() - 1)(*bib::Seed::random_engine());
       traj[i] = trajectory[r];
     }
  }
  
  void label_onpoltarget() {
    CHECK_GT(last_trajectory.size(), 0) << "Need at least one transition to label.";
    
    sample& last = last_trajectory[last_trajectory.size()-1];
    last.onpolicy_target = last.r;// Q-Val is just the final reward
    for (int i=last_trajectory.size()-2; i>=0; --i) {
      sample& t = last_trajectory[i];
      float reward = t.r;
      float target = last_trajectory[i+1].onpolicy_target;
      t.onpolicy_target = reward + gamma * target;
    }

  }
  
  void critic_update(std::deque<sample>& traj, uint iter){
    //compute \pi(s_{t+1})
    std::vector<double> all_next_states(traj.size() * nb_sensors);
    std::vector<double> all_states(traj.size() * nb_sensors);
    std::vector<double> all_actions(traj.size() * nb_motors);
    uint i=0;
    for (auto it : traj){
      std::copy(it.next_s.begin(), it.next_s.end(), all_next_states.begin() + i * nb_sensors);
      std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
      std::copy(it.a.begin(), it.a.end(), all_actions.begin() + i * nb_motors);
      i++;
    }

    auto all_next_actions = ann->computeOutBatch(all_next_states);
    //compute next q
    auto q_targets = qnn->computeOutVFBatch(all_next_states, *all_next_actions);
    delete all_next_actions;
    
    //adjust q_targets
    i=0;
    for (auto it : traj){
      if(it.goal_reached)
        q_targets->at(i) = it.r;
      else 
        q_targets->at(i) = it.r + gamma * q_targets->at(i);
      
      if(!pure_online)
        q_targets->at(i) = 0.5f * q_targets->at(i) + 0.5f*it.onpolicy_target;
      
      i++;
    }
    
    //Update critic
    qnn->InputDataIntoLayers(*qnn->getNN(), all_states.data(), all_actions.data(), q_targets->data(), NULL);
//     LOG_DEBUG("begin");
//     for(uint t=0;t<100;t++){
//       LOG_DEBUG(qnn->error());
    qnn->getSolver()->Step(1);
    
//     }
    
    delete q_targets;
    
    //usefull?
    qnn->ZeroGradParameters();
  }

  void actor_update_grad(std::deque<sample>& traj){
    
    std::vector<double> all_states(traj.size() * nb_sensors);
    uint i=0;
    for (auto it : traj){
      std::copy(it.s.begin(), it.s.end(), all_states.begin() + i * nb_sensors);
      i++;
    }
    
    //Update actor
    qnn->ZeroGradParameters();
    ann->ZeroGradParameters();
    
    auto all_actions_outputs = ann->computeOutBatch(all_states);
//     for(auto a : *all_actions_outputs)
//       if(a > 1.f || a < -1.f)
//         LOG_ERROR("aie");
    delete qnn->computeOutVFBatch(all_states, *all_actions_outputs);
    
    const auto q_values_blob = qnn->getNN()->blob_by_name(MLP::q_values_blob_name);
    double* q_values_diff = q_values_blob->mutable_cpu_diff();
    i=0;
    for (auto it : traj)
      q_values_diff[q_values_blob->offset(i++,0,0,0)] = -1.0f;
    qnn->getNN()->BackwardFrom(qnn->GetLayerIndex(MLP::q_values_layer_name));
    const auto critic_action_blob = qnn->getNN()->blob_by_name(MLP::actions_blob_name);
    double* action_diff = critic_action_blob->mutable_cpu_diff();
    
    for (uint n = 0; n < traj.size(); ++n) {
      for (uint h = 0; h < nb_motors; ++h) {
        int offset = critic_action_blob->offset(n,0,h,0);
        double diff = action_diff[offset];
        double output = all_actions_outputs->at(offset);
        double min = -1.0; 
        double max = 1.0;
        if (diff < 0) {
          diff *= (max - output) / (max - min);
        } else if (diff > 0) {
          diff *= (output - min) / (max - min);
        }
        action_diff[offset] = diff;
      }
    }
    
    // Transfer input-level diffs from Critic to Actor
    const auto actor_actions_blob = ann->getNN()->blob_by_name(MLP::actions_blob_name);
    actor_actions_blob->ShareDiff(*critic_action_blob);
    ann->getNN()->BackwardFrom(ann->GetLayerIndex("action_layer"));
    ann->getSolver()->ApplyUpdate();
    ann->getSolver()->set_iter(ann->getSolver()->iter() + 1);
    
    delete all_actions_outputs;
  }
  
  void end_episode() override {
    
    if(!pure_online){
      while(trajectory.size() + last_trajectory.size() > replay_memory)
        trajectory.pop_front();
      
      label_onpoltarget();
      auto it = trajectory.end();
      trajectory.insert(it, last_trajectory.begin(), last_trajectory.end());
      last_trajectory.clear();
    }
    
    if(!learning || trajectory.size() < kMinibatchSize)
      return;
    
    if(trajectory.size() != kMinibatchSize){
      kMinibatchSize = trajectory.size();
      qnn->increase_batchsize(kMinibatchSize);
      ann->increase_batchsize(kMinibatchSize);
    }
    
//     std::vector<sample> traj(kMinibatchSize);
//     sample_transition(traj);
    
    for(uint n=0;n<10; n++){
//       for(uint i=0; i<20 ; i++){
//         LOG_DEBUG(n << " " << i <<" " << std::setw(12) << std::fixed << std::setprecision(10) << qnn->error());
        critic_update(trajectory, 10);
//       }
      for(uint i=0; i<10 ; i++)
        actor_update_grad(trajectory);
    }
  }
  
  double criticEval(const std::vector<double>& perceptions, const std::vector<double>& actions) override {
      return qnn->computeOutVF(perceptions, actions);
  }
  
  arch::Policy<MLP>* getCopyCurrentPolicy() override {
        return new arch::Policy<MLP>(new MLP(*ann) , gaussian_policy ? arch::policy_type::GAUSSIAN : arch::policy_type::GREEDY, noise, decision_each);
  }

  void save(const std::string& path) override {
//      ann->save(path+".actor");
//      qnn->save(path+".critic");
//      bib::XMLEngine::save<>(trajectory, "trajectory", "trajectory.data");
  }

  void load(const std::string& path) override {
//     ann->load(path+".actor");
//     qnn->load(path+".critic");
  }

 protected:
  void _display(std::ostream& out) const override {
    out << std::setw(12) << std::fixed << std::setprecision(10) << sum_weighted_reward << " " << std::setw(8) <<
    std::fixed << std::setprecision(5) << noise << " " << trajectory.size() << " " << ann->weight_l1_norm() << " " 
    << std::fixed << std::setprecision(7) << qnn->error();
  }

  void _dump(std::ostream& out) const override {
    out <<" " << std::setw(25) << std::fixed << std::setprecision(22) <<
        sum_weighted_reward << " " << std::setw(8) << std::fixed <<
        std::setprecision(5) << trajectory.size() ;
  }
  

 private:
  uint nb_sensors;

  double noise;
  bool gaussian_policy;
  std::vector<uint>* hidden_unit_q;
  std::vector<uint>* hidden_unit_a;
  uint kMinibatchSize;
  uint replay_memory;
  
  bool learning, pure_online;

  std::shared_ptr<std::vector<double>> last_action;
  std::shared_ptr<std::vector<double>> last_pure_action;
  std::vector<double> last_state;

  std::deque<sample> trajectory;
  std::vector<sample> last_trajectory;
  
  MLP* ann;
  MLP* qnn;
};

#endif

