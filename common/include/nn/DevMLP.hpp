
#ifndef DevMLP_HPP
#define DevMLP_HPP

#include <caffe/layers/inner_product_layer.hpp>

#include "MLP.hpp"

//depends on CAFFE_CPU_ONLY

//DEBUG_DEVNN : to check if the output is really the same as the previous net
// #define DEBUG_DEVNN
//DEBUG_DEVNN_STOP : only print 
// #define DEBUG_DEVNN_STOP

#ifndef DEBUG_DEVNN
#undef DEBUG_DEVNN_STOP
#endif

class DevMLP : public MLP {
 public:

  DevMLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
         uint _kMinibatchSize, double decay_v, uint hidden_layer_type, uint batch_norm, bool _weighted_sample=false) :
        MLP(input, sensors, input-sensors, _kMinibatchSize, false, _weighted_sample, hiddens.size()), policy(false) {
    c = new constructor({hiddens, alpha, decay_v, hidden_layer_type, batch_norm, 0});
  }

//   policy
  DevMLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha, uint _kMinibatchSize,
         uint hidden_layer_type, uint last_layer_type, uint batch_norm, bool loss_layer=false) :
        MLP(sensors, sensors, motors, _kMinibatchSize, loss_layer, false, hiddens.size()), policy(true) {
    c = new constructor({hiddens, alpha, -1, hidden_layer_type, batch_norm, last_layer_type});

  }

  DevMLP(const MLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) :
    MLP(m, copy_solver, _phase) {
    //ok
  }

  virtual void exploit(boost::property_tree::ptree* pt, MLP* old) override {
    const uint task = 1;
    std::vector<uint> old_hiddens;

    bool fix_weights = pt->get<bool>("devnn.fix_weights");
    double init_multiplier = pt->get<double>("devnn.init_multiplier");
    bool start_same = pt->get<bool>("devnn.start_same");
    uint link_structure = pt->get<uint>("devnn.link_structure");
#ifdef DEBUG_DEVNN
    _old = old;
    init_multiplier = 0;
    start_same = true;
#endif
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

      string states_blob_name_new = net_param_old.layer(2).bottom(0)+".task1";
      string states_blob_name_old = net_param_old.layer(2).bottom(0)+".task0";

      if(old->size_sensors != size_sensors) {
        //split input
        SliceLayer(net_param_init, "slice_input", {states_blob_name}, {states_blob_name_old, states_blob_name_new},
                   boost::none, old->size_sensors);
      } else {
        states_blob_name_new = net_param_old.layer(2).bottom(0);
        states_blob_name_old = states_blob_name_new;
      }
      //add first inner
      int layer_index = 2;
//       bool layer
      while(net_param_old.layer(layer_index).type() != "InnerProduct") {
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(layer_index));
        if(net_param_old.layer(layer_index).type() == "BatchNorm") {
          lp->clear_param();
          //           lp->mutable_batch_norm_param()->set_use_global_stats(true); //cmaes crash
        }
        lp->set_bottom(0, states_blob_name_old);
        lp->set_top(0, states_blob_name_old);
        layer_index ++;
      }

      ASSERT(net_param_old.layer(layer_index).type() == "InnerProduct", "wrong fusion " << net_param_old.layer(2).type());
      lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(layer_index));
      lp->set_bottom(0, states_blob_name_old);
      if(fix_weights) {
        caffe::ParamSpec* lr_mult = lp->add_param();
        lr_mult->set_lr_mult(0.0f);
      }
      old_hiddens.push_back(lp->inner_product_param().num_output());
      layer_index++;

      //add everything else except last one
      for(; layer_index < net_param_old.layer_size() - 1 ; layer_index++) {
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(layer_index));
        if(lp->type() == "InnerProduct") {
          if(fix_weights) {
            caffe::ParamSpec* lr_mult = lp->add_param();
            lr_mult->set_lr_mult(0.0f);
          }
          old_hiddens.push_back(lp->inner_product_param().num_output());
        }
        if(net_param_old.layer(layer_index).type() == "BatchNorm") {
          lp->clear_param();
//           lp->mutable_batch_norm_param()->set_use_global_stats(true); //cmaes crash
        }
      }

      //change last output for concatenation
      lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(layer_index));
//       string actions_blob_name_old = net_param_old.layer(layer_index).top(0)+".task0";
      string actions_blob_name_new = net_param_old.layer(layer_index).top(0)+"."+task_name+std::to_string(task);
      lp->set_top(0, lp->name());

      std::string layer_name;
      ASSERT(old_hiddens.size() == c->hiddens.size(), "failed fusion ");
      if(link_structure != 0) {
//         for(uint i=1; i<=old_hiddens.size(); i++) {
        uint i = 1;
        if(link_structure & (1 << 1) && i==1 ) {
          layer_name = produce_name("rsh", i, task - 1);
          ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i, task - 1)}, {layer_name}, boost::none, {1,1,old_hiddens[i-1],1});
        }
        if(link_structure & (1 << 2) && i==1 ) {
          layer_name = produce_name("rsh", i + 1, task - 1);
          ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i + 1, task - 1)}, {layer_name}, boost::none, {1,1,old_hiddens[i],1});
        }
//         }
      }

      batch_norm_type bna = convertBN(c->batch_norm, c->hiddens.size());
      //produce new net
      std::string tower_top = Tower(net_param_init, states_blob_name_new, c->hiddens, bna,
                                    c->hidden_layer_type, link_structure, task, old->size_sensors == size_sensors);
      BatchNormTower(net_param_init, c->hiddens.size(), {tower_top}, {tower_top}, boost::none, bna, task);
      //concat everything
      std::vector<std::string> to_be_cc = {produce_name("ip", old_hiddens.size()+1, task-1), tower_top};
      if(link_structure & (1 << 0)) {
        to_be_cc.push_back(produce_name("ip", old_hiddens.size(), task-1));
      }
      ConcatLayer(net_param_init, "concat_last_layers", to_be_cc, {actions_blob_name_new}, boost::none, 1);

      layer_name = produce_name("ip", c->hiddens.size()+1, task);
      if(c->last_layer_type == 0)
        IPLayer(net_param_init, layer_name, {actions_blob_name_new}, {actions_blob_name}, boost::none, size_motors);
      else if(c->last_layer_type == 1) {
        IPLayer(net_param_init, layer_name, {actions_blob_name_new}, {actions_blob_name}, boost::none, size_motors);
        ReluLayer(net_param_init, produce_name("ip_func", c->hiddens.size()+1, task), {actions_blob_name}, {actions_blob_name},
                  boost::none);
      } else if(c->last_layer_type == 2) {
        IPLayer(net_param_init, layer_name, {actions_blob_name_new}, {actions_blob_name}, boost::none, size_motors);
        TanhLayer(net_param_init, produce_name("ip_func", c->hiddens.size()+1, task), {actions_blob_name}, {actions_blob_name},
                  boost::none);
      }

      if(add_loss_layer) {
        MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy2"},
                        boost::none, {kMinibatchSize, 1, size_motors, 1});
        EuclideanLossLayer(net_param_init, "loss", {actions_blob_name, targets_blob_name},
        {loss_blob_name}, boost::none);
      }

//       net_param_init.PrintDebugString();

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

      ASSERT(neural_net->blob_by_name(states_blob_name_old)->height() == old->neural_net->blob_by_name(
               states_blob_name)->height(), "failed fusion");
      ASSERT(add_loss_layer == false, "to be implemented");

//       adapt weight of last linear layer
      if(start_same) {
        std::string ip_last_layer_name = produce_name("ip", c->hiddens.size()+1, task);
        auto blob = neural_net->layer_by_name(ip_last_layer_name)->blobs()[0];
        auto blob_biais = neural_net->layer_by_name(ip_last_layer_name)->blobs()[1];
        double* weights = blob->mutable_cpu_data();
        double* weights_bias = blob_biais->mutable_cpu_data();

        for(int i=0; i < blob->count(); i++)
          weights[i] = weights[i] * init_multiplier;

        for(int i=0; i < blob_biais->count(); i++)
          weights_bias[i] = weights_bias[i] * init_multiplier;

        uint input_last_layer = old->size_motors + c->hiddens.back();
        if(link_structure & (1 << 0)) {
          input_last_layer += old_hiddens.back();
        }
        ASSERT((uint)blob->count() == (size_motors*input_last_layer),
               "failed fusion "<< blob->count() << " " << (size_motors*input_last_layer));

        uint y=0;
        for(uint i = 0; i<old->size_motors; i++) {
          weights[i*input_last_layer+y] = 1.f;
          y++;
        }
//         weights[0] = 1.f;
//         weights[(old->size_motors + c->hiddens.back())+1] = 1.f;//15
//         weights[3*(old->size_motors + c->hiddens.back())+2] = 1.f;//30
//         weights[4*(old->size_motors + c->hiddens.back())+3] = 1.f;//45

      }
    }
  }

  virtual double number_of_parameters() override {
    uint n = 0;
    caffe::Net<double>& net = *neural_net;
    ASSERT(net.learnable_params().size() == net.params_lr().size(), "failed");

    for (uint i = 0; i < net.learnable_params().size(); ++i) {
      if(net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        n += blob->count();
      }
    }

    return n;
  }

#ifdef DEBUG_DEVNN
#warning DEBUG_DEVNN
  MLP* _old;

  void copyWeightsFrom(const double*) override {
  }

#ifdef DEBUG_DEVNN_STOP
  std::vector<double>* computeOut(const std::vector<double>& states_batch) override {
    std::vector<double> substate(states_batch);
//     depends on the env :
//     substate.pop_back();
//     substate.pop_back();
//     substate.pop_back();
//     substate.pop_back();

    auto ac = _old->computeOut(substate);
//     return ac;
    bib::Logger::PRINT_ELEMENTS(*ac, "old action : ");
    delete ac;
    auto ac2 = MLP::computeOut(states_batch);
    bib::Logger::PRINT_ELEMENTS(*ac2, "new actions : ");

    auto actions_blob = neural_net->blob_by_name(produce_name("ip", c->hiddens.size()+1, 0));
    std::vector<double>* outputs = new std::vector<double>(actions_blob->count());
    for(int j=0; j < actions_blob->count(); j++)
      outputs->at(j) = actions_blob->data_at(0, j, 0, 0);
//     LOG_DEBUG("");
//     bib::Logger::PRINT_ELEMENTS(*outputs, "last layer of task0 : ");
//     exit(1);

    return ac2;
  }
#endif

#else
  virtual void copyWeightsFrom(const double* startx) override {
    uint index = 0;

    caffe::Net<double>& net = *neural_net;
    double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i)
      if(net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
#ifdef CAFFE_CPU_ONLY
        weights = blob->mutable_cpu_data();
#else
        switch (caffe::Caffe::mode()) {
        case caffe::Caffe::CPU:
          weights = blob->mutable_cpu_data();
          break;
        case caffe::Caffe::GPU:
          weights = blob->mutable_gpu_data();
          break;
        }
#endif
        for(int n=0; n < blob->count(); n++) {
          weights[n] = startx[index];
          index++;
        }
      }
  }
#endif

  virtual void copyWeightsTo(double* startx) override {
    uint index = 0;

    caffe::Net<double>& net = *neural_net;
    const double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) {
      if(net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
#ifdef CAFFE_CPU_ONLY
        weights = blob->cpu_data();
#else
        switch (caffe::Caffe::mode()) {
        case caffe::Caffe::CPU:
          weights = blob->cpu_data();
          break;
        case caffe::Caffe::GPU:
          weights = blob->gpu_data();
          break;
        }
#endif
        for(int n=0; n < blob->count(); n++) {
          startx[index] = weights[n];
          index++;
        }
      }
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



