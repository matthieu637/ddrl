#ifndef DODevMLP_HPP
#define DODevMLP_HPP

#include "MLP.hpp"
#include "bib/IniParser.hpp"

class DODevMLP : public MLP {
 public:

   DODevMLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
            uint _kMinibatchSize, double decay_v, uint hidden_layer_type, uint batch_norm, bool _weighted_sample=false) :
    MLP(input, sensors, hiddens, alpha, _kMinibatchSize, decay_v, hidden_layer_type, batch_norm, _weighted_sample) {

  }
    
  //   policy
  DODevMLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha, uint _kMinibatchSize,
            uint hidden_layer_type, uint last_layer_type, uint batch_norm, bool loss_layer=false) :
    MLP(sensors, hiddens, motors, alpha, _kMinibatchSize, hidden_layer_type, last_layer_type, batch_norm, loss_layer) {
    
  }

  DODevMLP(const MLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) : 
    DODevMLP(static_cast<const DODevMLP&>(m), copy_solver, _phase) {
  }
  
  DODevMLP(const DODevMLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) : 
    MLP(m, copy_solver, _phase), disable_st_control(m.disable_st_control), heuristic(m.heuristic) ,
    episode(m.episode), heuristic_devpoints_index(m.heuristic_devpoints_index) {
    st_control = new std::vector<uint>(*m.st_control);
    ac_control = new std::vector<uint>(*m.ac_control);
    if(heuristic == 1)
      heuristic_devpoints = new std::vector<uint>(*m.heuristic_devpoints);
    else if(heuristic == 2)
      heuristic_linearcoef = new std::vector<double>(*m.heuristic_linearcoef);
  }
  
  virtual ~DODevMLP(){
    if(st_control != nullptr)
      delete st_control;
    if(ac_control != nullptr)
      delete ac_control;
    if(heuristic_devpoints != nullptr)
      delete heuristic_devpoints;
    if(heuristic_linearcoef != nullptr)
      delete heuristic_linearcoef;
  }

  virtual void exploit(boost::property_tree::ptree* pt, MLP* actor) override {
    bool st_scale = pt->get<bool>("devnn.st_scale");
    bool ac_scale = pt->get<bool>("devnn.ac_scale");    
    uint st_probabilistic = pt->get<uint>("devnn.st_probabilistic");
    uint ac_probabilistic = pt->get<uint>("devnn.ac_probabilistic");
    st_control = bib::to_array<uint>(pt->get<std::string>("devnn.st_control"));
    ac_control = bib::to_array<uint>(pt->get<std::string>("devnn.ac_control"));
    heuristic = 0;
    try {
      heuristic = pt->get<uint>("devnn.heuristic");
      if((heuristic == 1 && st_probabilistic != 1) || (heuristic == 1 && ac_probabilistic != 1)){
        LOG_INFO("heuristic is 1 so probabilistic should be 1");
        exit(1);
      }
      LOG_INFO("catch heuristic " <<heuristic);
    } catch(boost::exception const& ) {
    }
    
    if(heuristic == 1){
      try {
        heuristic_devpoints = bib::to_array<uint>(pt->get<std::string>("devnn.heuristic_devpoints"));
        ASSERT((st_control->size() + ac_control->size()) <= heuristic_devpoints->size(), "st " << st_control->size() 
          << " " << ac_control->size() << " " << heuristic_devpoints->size() );
      } catch(boost::exception const& ) {
      }
    } else if(heuristic == 2){
      try {
        heuristic_linearcoef = bib::to_array<double>(pt->get<std::string>("devnn.heuristic_linearcoef"));
        ASSERT((st_control->size() + ac_control->size()) <= heuristic_linearcoef->size(), "st " << st_control->size() 
        << " " << ac_control->size() << " " << heuristic_linearcoef->size() );
      } catch(boost::exception const& ) {
      }
    }
    
    try {
      if(st_control->size() == 0 && pt->get<std::string>("devnn.st_control") == "None")
        disable_st_control = true;
    } catch(boost::exception const& ) {
    }

    if((ac_scale && ac_probabilistic != 0) || (st_scale && st_probabilistic != 0)) {
      LOG_ERROR("if not probabilistic then why scaling? of how much?");
      exit(1);
    }

    caffe::NetParameter net_param_init;
    caffe::NetParameter net_param_new;
    this->neural_net->ToProto(&net_param_init);
    this->neural_net->ToProto(&net_param_new);
    net_param_new.clear_layer();

    UpgradeNetBatchNorm(&net_param_new);
#ifndef NDEBUG
    if(MyNetNeedsUpgrade(net_param_new)) {
      LOG_ERROR("network need update");
      exit(1);
    }
#endif

//     net_param_init.PrintDebugString();
//     LOG_DEBUG("########################################################################################");
    uint nb_state_layer = 0;
    uint nb_action_layer = 0;
    bool input_action = false;
    bool state_dev_inserted = false;
    bool action_dev_inserted = false;
    for(caffe::LayerParameter& lp : *net_param_init.mutable_layer()) {
      uint b=0;
      bool state_capted_bottom = false;
      bool action_capted_bottom = false;
      for(const std::string& bot : lp.bottom()) {
        if(bot == MLP::states_blob_name && !disable_st_control) {
          *lp.mutable_bottom(b) = "devnn.states";
          nb_state_layer++;
          state_capted_bottom = true;
        } else if (bot == MLP::actions_blob_name && lp.name() != MLP::loss_blob_name) {
         *lp.mutable_bottom(b) = "devnn.actions";//change concat input
          action_capted_bottom = true;
        }
        b++;
      }
      b=0;
      for(const std::string& top : lp.top()) {
        if(top == MLP::actions_blob_name) {
          if(!input_action)
            input_action = lp.type() == "MemoryData";
          if(action_capted_bottom || !input_action)
            *lp.mutable_top(b) = "devnn.actions";
          nb_action_layer++;
        } else if(top == MLP::states_blob_name && state_capted_bottom && !disable_st_control){
          *lp.mutable_top(b) = "devnn.states";
        }
        b++;
      }
      
      if(nb_state_layer > 0 && !state_dev_inserted){
        caffe::DevelopmentalParameter* dp = addDevState(net_param_new, heuristic);
        dp->set_scale(st_scale);
        dp->set_probabilist(st_probabilistic);
        for (uint c : *st_control)
          dp->add_control(c);
        state_dev_inserted = true;
      }
      
      caffe::LayerParameter* lpc = net_param_new.add_layer();
      lpc->CopyFrom(lp);
      
      if(nb_action_layer > 0 && !action_dev_inserted && input_action){
        caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
        dp->set_scale(ac_scale);
        dp->set_probabilist(ac_probabilistic);
        for (uint c : *ac_control)
          dp->add_control(c);
        action_dev_inserted = true;
      }
    }
    
    if(nb_action_layer > 0 && !action_dev_inserted && !input_action){
      caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
      dp->set_scale(ac_scale);
      dp->set_probabilist(ac_probabilistic);
      for (uint c : *ac_control)
        dp->add_control(c);
      action_dev_inserted = true;
    }
    
#ifndef NDEBUG
    if(!disable_st_control)
      ASSERT(nb_state_layer >= 1, "check nb of state layer " << nb_state_layer);
#endif
    ASSERT(nb_action_layer >= 1, "check nb of action layer " << nb_action_layer);

//     net_param_new.PrintDebugString();
//     LOG_DEBUG("########################################################################################");

    net_param_new.set_force_backward(true);//mandatory for algorithm based on gradient (DDPG, ...)
    MLP::writeNN_struct(net_param_new, 1);
    
    caffe::SolverParameter solver_param(solver->param());
    solver_param.clear_net_param();
    caffe::NetParameter* net_param = solver_param.mutable_net_param();
    net_param->CopyFrom(net_param_new);

    delete solver;
    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();

    if(actor != nullptr){
      ASSERT(net_param_new.name() == "Critic", "net is not a critic " << net_param_new.name());
      ASSERT(actor->getNN()->name() == "Actor", "net is not an actor " << actor->getNN()->name());
      neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareData(*actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
      neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareDiff(*actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
      
      if(!disable_st_control){
        neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareData(*actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
        neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareDiff(*actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
      }
    }
    
    if(heuristic != 0){
      auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
      uint count = blob_ac->count();
      auto data = blob_ac->mutable_cpu_data();
      for(uint i=0;i<count;i++)
        data[i] = 0.0f;
      
      if(!disable_st_control){
        blob_ac = neural_net->layer_by_name("devnn_states")->blobs()[0];
        count = blob_ac->count();
        data = blob_ac->mutable_cpu_data();
        for(uint i=0;i<count;i++)
          data[i] = 0.0f;
      }
    }
  }

 private:
   caffe::DevelopmentalParameter* addDevState(caffe::NetParameter& net_param, uint heuristic) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, "devnn_states", "Developmental", {MLP::states_blob_name}, {"devnn.states"}, boost::none);
    if(heuristic != 0){
      caffe::ParamSpec* fixed_param_spec = layer.add_param();
      fixed_param_spec->set_lr_mult(0.f);
      fixed_param_spec->set_decay_mult(0.f);
    }
    return layer.mutable_developmental_param();
  }

  caffe::DevelopmentalParameter* addDevAction(caffe::NetParameter& net_param, bool input_action, uint heuristic) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    if(input_action)
      PopulateLayer(layer, "devnn_actions", "Developmental", {MLP::actions_blob_name}, {"devnn.actions"}, boost::none);
    else
      PopulateLayer(layer, "devnn_actions", "Developmental", {"devnn.actions"}, {MLP::actions_blob_name}, boost::none);
    if(heuristic != 0){
      caffe::ParamSpec* fixed_param_spec = layer.add_param();
      fixed_param_spec->set_lr_mult(0.f);
      fixed_param_spec->set_decay_mult(0.f);
    }
    return layer.mutable_developmental_param();
  }
  
  void develop(){
    if(heuristic == 1){
      while(heuristic_devpoints_index < heuristic_devpoints->size() && episode == heuristic_devpoints->at(heuristic_devpoints_index)){
        if(heuristic_devpoints_index < st_control->size()){
          auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
          auto data = blob_st->mutable_cpu_data();
          data[heuristic_devpoints_index] = 1.f;
          LOG_INFO("dev point st " << st_control->at(heuristic_devpoints_index) );
        } else if(heuristic_devpoints_index < st_control->size() + ac_control->size()) {
          auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
          auto data = blob_ac->mutable_cpu_data();
          data[heuristic_devpoints_index - st_control->size()] = 1.f;
          LOG_INFO("dev point ac " << ac_control->at(heuristic_devpoints_index - st_control->size()) );
        }
        
        heuristic_devpoints_index++;
      }
    } else if(heuristic == 2) {
      if(!disable_st_control){
        auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
        auto data_st = blob_st->mutable_cpu_data();
        for(uint i=0; i < st_control->size() ; i++){
          data_st[i] = ((double)episode) * heuristic_linearcoef->at(i);
        }
      }
      
      auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
      auto data = blob_ac->mutable_cpu_data();
      for(uint i=0; i < ac_control->size() ; i++){
        data[i] = ((double)episode) * heuristic_linearcoef->at(i + st_control->size());
      }
    }
  }
  
public:
//   Used by CMA-ES
  void copyWeightsFrom(const double* startx, bool ignore_null_lr) override {
    develop();
    
    MLP::copyWeightsFrom(startx, ignore_null_lr);
    episode++;
  }
  
//   Used by others
  void inform(uint episode_){
    episode = episode_;
    develop();
  }
  
  void soft_update(const MLP& from, double tau) override {
    auto net_from = from.neural_net;
    auto net_to = neural_net;
    
    const auto& from_params = net_from->params();
    const auto& to_params = net_to->params();
    CHECK_EQ(from_params.size(), to_params.size());
    CHECK_EQ(neural_net->params_lr().size(), to_params.size());
    
    for (uint i = 0; i < from_params.size(); ++i) {
      if(neural_net->params_lr()[i] != 0.0f){
        auto& from_blob = from_params[i];
        auto& to_blob = to_params[i];
        caffe::caffe_cpu_axpby(from_blob->count(), tau, from_blob->cpu_data(),
                              (1.f-tau), to_blob->mutable_cpu_data());
      }
    }
  }
  
  virtual void actor_backward() override {
    int layer = get_layer_index("devnn_actions");
    
    ASSERT(layer != -1 , "failed to found layer for backward");
    
    neural_net->BackwardFrom(layer);
  }
  
private:
  bool disable_st_control = false;
  uint heuristic = 0;
  uint episode = 0;
  uint heuristic_devpoints_index = 0;
  std::vector<uint>* heuristic_devpoints = nullptr;
  std::vector<double>* heuristic_linearcoef = nullptr;
  std::vector<uint>* st_control = nullptr;
  std::vector<uint>* ac_control = nullptr;
};

#endif




