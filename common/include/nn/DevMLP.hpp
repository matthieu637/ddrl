
#ifndef DevMLP_HPP
#define DevMLP_HPP

#include "MLP.hpp"

//depends on CAFFE_CPU_ONLY

class DevMLP : public MLP {
 public:

  DevMLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
         uint _kMinibatchSize, double decay_v, uint hidden_layer_type, uint batch_norm, bool _weighted_sample=false) :
    MLP(input, sensors, input-sensors, _kMinibatchSize, false, _weighted_sample), policy(false) {
    c = new constructor({hiddens, alpha, decay_v, hidden_layer_type, batch_norm, 0});
  }

//   policy
  DevMLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha, uint _kMinibatchSize,
         uint hidden_layer_type, uint last_layer_type, uint batch_norm, bool loss_layer=false) :
    MLP(sensors, sensors, motors, _kMinibatchSize, loss_layer, false), policy(true) {
    c = new constructor({hiddens, alpha, -1, hidden_layer_type, batch_norm, last_layer_type});

  }

  DevMLP(const MLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) :
    MLP(m, copy_solver, _phase) {
    //ok
  }

  virtual void exploit(boost::property_tree::ptree*, MLP* old) override {
    caffe::NetParameter net_param_old;
    old->getNN()->ToProto(&net_param_old);

    if(policy) {
      caffe::NetParameter net_param_init;
      net_param_init.set_name("Actor");
      net_param_init.set_force_backward(true);

      //copy input layer and increase dimension
      caffe::LayerParameter* lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(0));
      caffe::MemoryDataParameter* mdataparam = lp->mutable_memory_data_param();
      mdataparam->set_height(size_sensors);

      //copy silent layer
      lp = net_param_init.add_layer();
      ASSERT(net_param_old.layer(1).type() == "Silence", "wrong fusion");
      lp->CopyFrom(net_param_old.layer(1));

      //split input
      string states_blob_name_new = net_param_old.layer(2).bottom(0)+"_new";
      string states_blob_name_old = net_param_old.layer(2).bottom(0)+"_old";
      SliceLayer(net_param_init, "slice_input", {states_blob_name}, {states_blob_name_new, states_blob_name_old},
                 boost::none, size_sensors - old->size_sensors);

      //add first inner
      ASSERT(net_param_old.layer(2).type() == "InnerProduct", "wrong fusion");
      lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(2));
      lp->set_bottom(0, states_blob_name_old);

      //add everything else
      int i=3;
      for(; i < net_param_old.layer_size() - 1 ; i++) {
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(i));
      }

      //change last output for concatenation
      lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(i));
      string actions_blob_name_old = net_param_old.layer(i).top(0)+"_old";
      string actions_blob_name_new = net_param_old.layer(i).top(0)+"_new";
      lp->set_top(0, actions_blob_name_old);

      //produce new net
      std::string tower_top = Tower(net_param_init, "sub1", states_blob_name_new, c->hiddens, c->batch_norm,
                                    c->hidden_layer_type);
      if(c->batch_norm ==4) {
        std::string layer_name2 = "final_bn";
        BatchNormLayer(net_param_init, layer_name2, {tower_top}, {layer_name2}, boost::none);
        std::string layer_name3 = "final_sc";
        tower_top = tower_top+"_sc";
        ScaleLayer(net_param_init, layer_name3, {layer_name2}, {tower_top}, boost::none, 1);
      }

      //concat everything
      ConcatLayer(net_param_init, "concat_last_layers", {tower_top, actions_blob_name_old}, {actions_blob_name_new},
                  boost::none, 1);

      if(c->last_layer_type == 0)
        IPLayer(net_param_init, "action_layer", {actions_blob_name_new}, {actions_blob_name}, boost::none, size_motors);
      else if(c->last_layer_type == 1) {
        IPLayer(net_param_init, "action_layer_ip", {actions_blob_name_new}, {"last_relu"}, boost::none, size_motors);
        ReluLayer(net_param_init, "action_layer", {"last_relu"}, {actions_blob_name}, boost::none);
      } else if(c->last_layer_type == 2) {
        IPLayer(net_param_init, "action_layer_ip", {actions_blob_name_new}, {"last_tanh"}, boost::none, size_motors);
        TanhLayer(net_param_init, "action_layer", {"last_tanh"}, {actions_blob_name}, boost::none);
      }

      if(add_loss_layer) {
        MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy2"},
                        boost::none, {kMinibatchSize, 1, size_motors, 1});
        EuclideanLossLayer(net_param_init, "loss", {actions_blob_name, targets_blob_name},
        {loss_blob_name}, boost::none);
      }

      caffe::SolverParameter solver_param;
      caffe::NetParameter* net_param = solver_param.mutable_net_param();
      net_param->CopyFrom(net_param_init);

      solver_param.set_type("Adam");
      solver_param.set_max_iter(10000000);
      solver_param.set_base_lr(c->alpha);
      solver_param.set_lr_policy("fixed");
      solver_param.set_snapshot_prefix("actor");
      solver_param.set_clip_gradients(10);

      solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
      neural_net = solver->net();
      
      ASSERT(neural_net->blob_by_name(states_blob_name_old)->height() == old->neural_net->blob_by_name(states_blob_name)->height(), "failed fusion");
      ASSERT(add_loss_layer == false, "to be implemented");
      ASSERT(c->batch_norm == 0, "to be implemented");
      //TODO:adapt weight of last linear layer
    }
  }

 public:
  virtual ~DevMLP() {
    if(c != nullptr)
      delete c;
  }

 private:
  struct constructor {
    const std::vector<uint>& hiddens;
    double alpha;
    double decay_v;
    uint hidden_layer_type;
    uint batch_norm;
    uint last_layer_type;
  };

  bool policy;
  constructor* c = nullptr;

};

#endif  // DevMLP_HPP


