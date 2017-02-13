
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
    
    if(link_structure == 8 && start_same){
      LOG_ERROR("if link_structure == 8 then start_same should be false");
      exit(1);
    }
    
    if(link_structure == 9 && !start_same){
      LOG_ERROR("if link_structure == 9 then start_same should be true");
      exit(1);
    }
    
    if(link_structure == 9 && fix_weights){
      LOG_ERROR("if link_structure == 9 then fix_weights should be false");
      exit(1);
    }
    
    
#ifdef DEBUG_DEVNN
    _old = old;
    init_multiplier = 0;
    start_same = true;
#endif
    caffe::NetParameter net_param_old;
    old->getNN()->ToProto(&net_param_old);
    
//     net_param_old.PrintDebugString();
//     LOG_DEBUG("########################################################################################");

      caffe::NetParameter net_param_init;
      if(!policy)
        net_param_init.set_name("Critic");
      else
        net_param_init.set_name("Actor");
      net_param_init.set_force_backward(true);

      //copy input layer and increase dimension
      caffe::LayerParameter* lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(0));
      caffe::MemoryDataParameter* mdataparam = lp->mutable_memory_data_param();
      mdataparam->set_height(size_sensors);
      const uint old_batch_size = mdataparam->batch_size();
      
      int layer_index = 1; 
      if(!policy){
        layer_index = 3;
        
        //action layer
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(1));
        mdataparam = lp->mutable_memory_data_param();
        mdataparam->set_height(size_motors);
        
        //target layer
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(2));
      }

      //copy silent layer
      lp = net_param_init.add_layer();
      ASSERT(net_param_old.layer(layer_index).type() == "Silence", "wrong fusion 1) " << net_param_old.layer(layer_index).type());
      lp->CopyFrom(net_param_old.layer(layer_index));
      layer_index++;
      
      string states_blob_name_new = net_param_old.layer(layer_index).bottom(0)+".task1";
      string states_blob_name_old = net_param_old.layer(layer_index).bottom(0)+".task0";
      if(old->size_sensors != size_sensors && link_structure != 9) {
        //split input
        SliceLayer(net_param_init, "slice_input", {states_blob_name}, {states_blob_name_old, states_blob_name_new},
                   boost::none, old->size_sensors);
      } else {
        states_blob_name_new = net_param_old.layer(layer_index).bottom(0);
        states_blob_name_old = states_blob_name_new;
      }
      
      string actions_blob_name_new;
      string actions_blob_name_old;
      
      string states_actions_blob_name_new;
      string states_actions_blob_name_old;
      if(!policy && link_structure != 9){
        actions_blob_name_new = net_param_old.layer(layer_index).bottom(1)+".task1";
        actions_blob_name_old = net_param_old.layer(layer_index).bottom(1)+".task0";
        
        states_actions_blob_name_new = states_actions_blob_name+"."+task_name+std::to_string(task);
        states_actions_blob_name_old = states_actions_blob_name+"."+task_name+std::to_string(task-1);
        
        if(old->size_motors != size_motors) {
          //split input
          SliceLayer(net_param_init, "slice_input2", {actions_blob_name}, {actions_blob_name_old, actions_blob_name_new},
                    boost::none, old->size_motors);
        } else {
          states_blob_name_new = net_param_old.layer(layer_index).bottom(1);
          states_blob_name_old = states_blob_name_new;
        }
      } else if(!policy){
        actions_blob_name_new = net_param_old.layer(layer_index).bottom(1);
        actions_blob_name_old = net_param_old.layer(layer_index).bottom(1);
        
        states_actions_blob_name_new = states_actions_blob_name;
        states_actions_blob_name_old = states_actions_blob_name;
      }
      
      //add first inner
      while(net_param_old.layer(layer_index).type() != "InnerProduct") {
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(layer_index));
        if(net_param_old.layer(layer_index).type() == "BatchNorm") {
          lp->clear_param();
          //           lp->mutable_batch_norm_param()->set_use_global_stats(true); //cmaes crash
        }
        lp->set_bottom(0, states_blob_name_old);
        lp->set_top(0, states_blob_name_old);
        
        if(!policy){
          lp->set_bottom(1, actions_blob_name_old);
          lp->set_top(0, states_actions_blob_name_old);
        }
        
        layer_index ++;
      }
      
      if(!policy && link_structure != 9){
        ConcatLayer(net_param_init, "concat2", {states_blob_name_new, actions_blob_name_new}, {states_actions_blob_name_new}, boost::none, 2);
      }

      ASSERT(net_param_old.layer(layer_index).type() == "InnerProduct", "wrong fusion " << net_param_old.layer(2).type());
      lp = net_param_init.add_layer();
      lp->CopyFrom(net_param_old.layer(layer_index));
      lp->set_bottom(0, states_blob_name_old);
      if(!policy)
        lp->set_bottom(0, states_actions_blob_name_old);
      if(fix_weights) {
        caffe::ParamSpec* lr_mult = lp->add_param();
        lr_mult->set_lr_mult(0.0f);
      }
      if(link_structure == 9 && size_sensors != old->size_sensors && policy){
        lp->mutable_blobs(0)->mutable_shape()->set_dim(1, size_sensors);
        std::vector<double> save(old->size_sensors * lp->blobs(0).shape().dim(0));
        for(uint i=0; i < old->size_sensors * lp->blobs(0).shape().dim(0); i++){
          save[i] = lp->blobs(0).double_data(i);
          lp->mutable_blobs(0)->set_double_data(i, 0);
        }
        for(uint i=0;i < (size_sensors - old->size_sensors) * lp->blobs(0).shape().dim(0) ;i++)
          lp->mutable_blobs(0)->add_double_data(init_multiplier * bib::Seed::gaussianRand<double>(0, lp->inner_product_param().weight_filler().std() ));
//           lp->mutable_blobs(0)->add_double_data(0);
        uint y=0;
        for(uint i=0;i < size_sensors * lp->blobs(0).shape().dim(0) ;i++) {
          if((i % size_sensors) < old->size_sensors )
          lp->mutable_blobs(0)->set_double_data(i, save[y++]);
        }
      } else if(link_structure == 9 && !policy){
        uint sum_ms = size_motors + size_sensors;
        uint sum_ms_old = old->size_motors + old->size_sensors;
        
        lp->mutable_blobs(0)->mutable_shape()->set_dim(1, sum_ms);
        std::vector<double> save(sum_ms_old * lp->blobs(0).shape().dim(0));
        for(uint i=0; i < sum_ms_old * lp->blobs(0).shape().dim(0); i++){
          save[i] = lp->blobs(0).double_data(i);
          lp->mutable_blobs(0)->set_double_data(i, 0);
        }
        for(uint i=0;i < (sum_ms - sum_ms_old) * lp->blobs(0).shape().dim(0) ;i++)
          lp->mutable_blobs(0)->add_double_data(init_multiplier * bib::Seed::gaussianRand<double>(0, lp->inner_product_param().weight_filler().std() ));
//           lp->mutable_blobs(0)->add_double_data(0);
        
        uint y=0;
        for(uint i=0;i < sum_ms * lp->blobs(0).shape().dim(0) ;i++) {
          uint li = i % sum_ms;
          if(li < old->size_sensors || (li >= size_sensors && li < (size_sensors + old->size_motors)))
            lp->mutable_blobs(0)->set_double_data(i, save[y++]);
        }
      }
      old_hiddens.push_back(lp->inner_product_param().num_output());
      layer_index++;
      
      int last_one_layer = net_param_old.layer_size() - 1;
      if(!policy)
        last_one_layer--;
      //add everything else except last one
      for(; layer_index < last_one_layer ; layer_index++) {
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
      actions_blob_name_new = net_param_old.layer(layer_index).top(0)+"."+task_name+std::to_string(task);
      lp->set_top(0, lp->name());
      
      std::string layer_name;
      ASSERT(old_hiddens.size() == c->hiddens.size(), "failed fusion 2) " << old_hiddens.size() << " " << c->hiddens.size());
      if(link_structure <= 7) {
        uint i = 1;
        if(link_structure & (1 << 1) && i==1 ) {
          layer_name = produce_name("rsh", i, task - 1);
          ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i, task - 1)}, {layer_name}, boost::none, {old_batch_size,1,old_hiddens[i-1],1});
        }
        if(link_structure & (1 << 2) && i==1 ) {
          layer_name = produce_name("rsh", i + 1, task - 1);
          ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i + 1, task - 1)}, {layer_name}, boost::none, {old_batch_size,1,old_hiddens[i],1});
        }
      } else if(link_structure == 8) {
        std::vector<std::string> to_be_cc;
        if(policy){
          to_be_cc.push_back(states_blob_name_new);
          to_be_cc.push_back(states_blob_name_old);
        } else {
          to_be_cc.push_back(states_actions_blob_name_new);
          to_be_cc.push_back(states_actions_blob_name_old);
        }
        
        for(uint i=1; i < old_hiddens.size() + 2; i++) {
          layer_name = produce_name("rsh", i, task - 1);

          if(i - 1 < old_hiddens.size())
            ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i, task - 1)}, {layer_name}, boost::none, {old_batch_size,1,old_hiddens[i-1],1});
          else if(policy)
            ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i, task - 1)}, {layer_name}, boost::none, {old_batch_size,1,old->size_motors,1});
          else //!policy
            ReshapeLayer(net_param_init, layer_name, {produce_name("ip", i, task - 1)}, {layer_name}, boost::none, {old_batch_size,1,1,1});

          to_be_cc.push_back(layer_name);
        }
        layer_name = produce_name("cc", 1, task);
        ConcatLayer(net_param_init, layer_name, to_be_cc, {layer_name}, boost::none, 2);
        states_blob_name_new = layer_name;
      }
      
      if(link_structure != 9){
        string tower_input = states_blob_name_new;
        if(!policy){
          if(link_structure <= 7)
            tower_input = states_actions_blob_name_new;
          else //link_structure == 8
            tower_input = states_blob_name_new; //just edited before
        }
        
        batch_norm_type bna = convertBN(c->batch_norm, c->hiddens.size());
        //produce new net
        std::string tower_top = Tower(net_param_init, tower_input, c->hiddens, bna,
                                      c->hidden_layer_type, link_structure == 8 ? 0 : link_structure,
                                      task, old->size_sensors == size_sensors, policy);
        BatchNormTower(net_param_init, c->hiddens.size(), {tower_top}, {tower_top}, boost::none, bna, task);
      
        std::vector<std::string> final_layer;
        if(! (link_structure & (1 << 3))) { // link_structure in [0 ; 7]
          //concat everything
          std::vector<std::string> to_be_cc = {produce_name("ip", old_hiddens.size()+1, task-1), tower_top};
          if(link_structure & (1 << 0)) {
            to_be_cc.push_back(produce_name("ip", old_hiddens.size(), task-1));
          }

          ConcatLayer(net_param_init, "concat_last_layers", to_be_cc, {actions_blob_name_new}, boost::none, 1);
          final_layer.push_back(actions_blob_name_new);
        } else if(link_structure == 8) {
          final_layer.push_back(produce_name("ip", c->hiddens.size(), task));
        }
      
        string last_layer_name = actions_blob_name;
        uint num_output = size_motors;
        if(!policy) {
          last_layer_name = q_values_blob_name;
          num_output = 1;
        }
        
        layer_name = produce_name("ip", c->hiddens.size()+1, task);
        if(c->last_layer_type == 0)
          IPLayer(net_param_init, layer_name, final_layer, {last_layer_name}, boost::none, num_output);
        else if(c->last_layer_type == 1) {
          IPLayer(net_param_init, layer_name, final_layer, {last_layer_name}, boost::none, num_output);
          ReluLayer(net_param_init, produce_name("ip_func", c->hiddens.size()+1, task), {last_layer_name}, {last_layer_name},
                    boost::none);
        } else if(c->last_layer_type == 2) {
          IPLayer(net_param_init, layer_name, final_layer, {last_layer_name}, boost::none, num_output);
          TanhLayer(net_param_init, produce_name("ip_func", c->hiddens.size()+1, task), {last_layer_name}, {last_layer_name},
                    boost::none);
        }
      
        if(add_loss_layer) {
          MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy2"},
                          boost::none, {kMinibatchSize, 1, num_output, 1});
          EuclideanLossLayer(net_param_init, "loss", {last_layer_name, targets_blob_name},
          {loss_blob_name}, boost::none);
        }
      } else { //link_structure == 9
        if(policy){
          lp->mutable_inner_product_param()->set_num_output(size_motors);
          lp->mutable_blobs(0)->mutable_shape()->set_dim(0, size_motors);
          for(uint i=0;i < (size_motors - old->size_motors) * lp->blobs(0).shape().dim(1) ;i++)
            lp->mutable_blobs(0)->add_double_data(init_multiplier * bib::Seed::gaussianRand<double>(0, lp->inner_product_param().weight_filler().std() ));        
          
          lp->mutable_blobs(1)->mutable_shape()->set_dim(0, size_motors);
          for(uint i=0;i < size_motors - old->size_motors;i++)
            lp->mutable_blobs(1)->add_double_data(init_multiplier * bib::Seed::gaussianRand<double>(0, lp->inner_product_param().weight_filler().std()) );
          lp->set_top(0, actions_blob_name);
        } else {
          lp->set_top(0, q_values_blob_name);
        }
      }
      
      if(!policy){
        lp = net_param_init.add_layer();
        lp->CopyFrom(net_param_old.layer(layer_index+1));
      }
//       net_param_old.PrintDebugString();
//       LOG_DEBUG("#############################################");
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
      LOG_INFO("nb param : "  << number_of_parameters() << " : " << link_structure );
      
#ifndef NDEBUG
      if(link_structure != 9)
        ASSERT(neural_net->blob_by_name(states_blob_name_old)->height() == old->neural_net->blob_by_name(
               states_blob_name)->height(), "failed fusion");
#endif
      ASSERT(add_loss_layer == false, "to be implemented");

//       adapt weight of last linear layer
      if(start_same && link_structure != 9) {
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
        if(!policy)
          input_last_layer = 1 + c->hiddens.back();
        
        if(link_structure & (1 << 0)) {
          input_last_layer += old_hiddens.back();
        }
#ifndef NDEBUG
        if(policy)
          ASSERT((uint)blob->count() == (size_motors*input_last_layer),
               "failed fusion "<< blob->count() << " " << (size_motors*input_last_layer));
        else
          ASSERT((uint)blob->count() == (1*input_last_layer),
                 "failed fusion "<< blob->count() << " " << (1*input_last_layer));
#endif
        uint y=0;
        if(policy)
          for(uint i = 0; i<old->size_motors; i++) {
            weights[i*input_last_layer+y] = 1.f;
            y++;
          }
        else
          for(uint i = 0; i < 1; i++) {
            weights[i*input_last_layer+y] = 1.f;
            y++;
          }
          
//         weights[0] = 1.f;
//         weights[(old->size_motors + c->hiddens.back())+1] = 1.f;//15
//         weights[3*(old->size_motors + c->hiddens.back())+2] = 1.f;//30
//         weights[4*(old->size_motors + c->hiddens.back())+3] = 1.f;//45

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
//     depends on the env :(and predev)
    substate.pop_back();
    substate.pop_back();
    substate.pop_back();
    substate.pop_back();

    auto ac = _old->computeOut(substate);
    auto ac2 = MLP::computeOut(states_batch);
//     bib::Logger::PRINT_ELEMENTS(*ac, "old action : ");
//     delete ac;
//     bib::Logger::PRINT_ELEMENTS(*ac2, "new actions : ");

    return ac2;
  }
  
  double computeOutVF(const std::vector<double>& states_batch, const std::vector<double>& motors_batch) override {
    bib::Logger::PRINT_ELEMENTS(states_batch, "state : ");
    bib::Logger::PRINT_ELEMENTS(motors_batch, "action : ");
    
    std::vector<double> substate(states_batch);
    std::vector<double> subaction(motors_batch);
//     depends on the env : (and predev)
//     substate.pop_back();
//     substate.pop_back();
//     substate.pop_back();
//     substate.pop_back();
    
    subaction.pop_back();
    subaction.pop_back();
    
    double q1 = _old->computeOutVF(substate, subaction);
    double q2 = MLP::computeOutVF(states_batch, motors_batch);
    LOG_DEBUG(q1);
    LOG_DEBUG(q2 << "\n");
    
    return q2;
  }
  
  std::vector<double>* computeOutVFBatch(std::vector<double>& sensors, std::vector<double>& motors) override {
    caffe::NetParameter net_param_old;
    _old->neural_net->ToProto(&net_param_old);
    net_param_old.PrintDebugString();
    LOG_DEBUG("#############################################");
    caffe::NetParameter net_param_old2;
    neural_net->ToProto(&net_param_old2);
    net_param_old2.PrintDebugString();
    
    
    
    auto outputs = new std::vector<double>(kMinibatchSize);
    
    for (uint n = 0; n < kMinibatchSize; ++n){
      vector<double>::const_iterator s_fisrt = sensors.begin() + n*size_sensors;
      vector<double>::const_iterator s_end = sensors.begin() + (n+1)*size_sensors;
      vector<double> ns(s_fisrt, s_end);
      
      vector<double>::const_iterator a_fisrt = motors.begin() + n*size_motors;
      vector<double>::const_iterator a_end = motors.begin() + (n+1)*size_motors;
      vector<double> ac(a_fisrt, a_end);
      
      outputs->at(n) = this->computeOutVF(ns, ac);
      
      exit(1);
    }
    
    return outputs;
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




