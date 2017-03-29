#ifndef DODevMLP_HPP
#define DODevMLP_HPP

#include "MLP.hpp"
#include "bib/IniParser.hpp"

class DODevMLP : public MLP {
 public:

  using MLP::MLP;

  virtual void exploit(boost::property_tree::ptree* pt, MLP* actor) override {
    bool st_scale = pt->get<bool>("devnn.st_scale");
    bool ac_scale = pt->get<bool>("devnn.ac_scale");    
    uint st_probabilistic = pt->get<uint>("devnn.st_probabilistic");
    uint ac_probabilistic = pt->get<uint>("devnn.ac_probabilistic");
    std::vector<uint>* st_control = bib::to_array<uint>(pt->get<std::string>("devnn.st_control"));
    std::vector<uint>* ac_control = bib::to_array<uint>(pt->get<std::string>("devnn.ac_control"));
    uint heuristic = 0;
    try {
      heuristic = pt->get<uint>("devnn.heuristic");
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
      for(const std::string& bot : lp.bottom()) {
        if(bot == MLP::states_blob_name) {
          *lp.mutable_bottom(b) = "devnn.states";
          nb_state_layer++;
          state_capted_bottom = true;
        } else if (bot == MLP::actions_blob_name && lp.name() != MLP::loss_blob_name) {
          *lp.mutable_bottom(b) = "devnn.actions";
          input_action = true;
          nb_action_layer++;
        }
        b++;
      }
      b=0;
      for(const std::string& top : lp.top()) {
        if(top == MLP::actions_blob_name) {
          *lp.mutable_top(b) = "devnn.actions";
          ASSERT(!input_action, "action both in input and output mode");
          input_action=false;
          nb_action_layer++;
        } else if(top == MLP::states_blob_name && state_capted_bottom){
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
      
      if(nb_action_layer > 0 && input_action && !action_dev_inserted){
        caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
        dp->set_scale(ac_scale);
        dp->set_probabilist(ac_probabilistic);
        for (uint c : *ac_control)
          dp->add_control(c);
        action_dev_inserted = true;
      }
      
      caffe::LayerParameter* lpc = net_param_new.add_layer();
      lpc->CopyFrom(lp);
      
      if(nb_action_layer > 0 && !input_action && !action_dev_inserted){
        caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
        dp->set_scale(ac_scale);
        dp->set_probabilist(ac_probabilistic);
        for (uint c : *ac_control)
          dp->add_control(c);
        action_dev_inserted = true;
      }
    }
    ASSERT(nb_state_layer >= 1, "check nb of state layer " << nb_state_layer);
    ASSERT(nb_action_layer >= 1, "check nb of action layer " << nb_action_layer);

//     net_param_new.PrintDebugString();
//     LOG_DEBUG("########################################################################################");

    caffe::SolverParameter solver_param(solver->param());
    solver_param.clear_net_param();
    caffe::NetParameter* net_param = solver_param.mutable_net_param();
    net_param->CopyFrom(net_param_new);

    delete solver;
    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();

    if(actor != nullptr){
      ASSERT(net_param_new.name() == "Critic", "net is not a critic " << net_param_new.name());
      neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareData(*actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
      neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareDiff(*actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
      
      neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareData(*actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
      neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareDiff(*actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
    }
    
    delete st_control;
    delete ac_control;
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
};

#endif
