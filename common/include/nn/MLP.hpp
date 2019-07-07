
#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <functional>
#include <thread>

#include "bib/Logger.hpp"
#include <bib/Converger.hpp>
#include <bib/Seed.hpp>
#include <bib/Utils.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/batch_norm_layer.hpp>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>

//depends on CAFFE_CPU_ONLY


class MLP {
 public:
  friend class DevMLP;
  friend class DODevMLP;

  // Layer Names
  static const std::string state_input_layer_name;
  static const std::string action_input_layer_name;
  static const std::string target_input_layer_name;
  static const std::string wsample_input_layer_name;

  // Blob names
  static const std::string states_blob_name;
  static const std::string actions_blob_name;
  static const std::string states_actions_blob_name;
  static const std::string targets_blob_name;
  static const std::string wsample_blob_name;
  static const std::string q_values_blob_name;
  static const std::string loss_blob_name;

  static const std::string task_name;

  typedef enum {none, first, all_except_last, all, first_except_action, very_all} batch_norm_arch;
  struct batch_norm_type {
    batch_norm_arch arch;
    bool with_scale;
    bool with_scale_bias;
    uint max_rank;
  };

  inline batch_norm_type convertBN(uint bn, uint max_rank) {
    batch_norm_type r;
    r.with_scale_bias = bn & (1 << 0);
    r.with_scale = bn & (1 << 1);
    r.max_rank = max_rank;
    if ((bn & (1 << 4)) && !(bn & (1 << 2)))
      r.arch = batch_norm_arch::first_except_action;
    else if (bn & (1 << 4))
      r.arch = batch_norm_arch::very_all;
    else if((bn & (1 << 2)) && (bn & (1 << 3)))
      r.arch = batch_norm_arch::all;
    else if((bn & (1 << 2)))
      r.arch = batch_norm_arch::first;
    else if((bn & (1 << 3)))
      r.arch = batch_norm_arch::all_except_last;
    else
      r.arch = batch_norm_arch::none;
#ifndef NDEBUG
    if(r.with_scale_bias)
      ASSERT(r.with_scale, "try to scale with bias without scaling");
    if(r.arch == batch_norm_arch::none) {
      ASSERT(!r.with_scale_bias, "try to scale with bias without bn");
      ASSERT(!r.with_scale, "try to scale without bn");
    }
#endif
    return r;
  }

  constexpr static auto kFixedDim = 1;

//   
//   CRITIC NET
//   
  MLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
      uint _kMinibatchSize, double decay_v, uint hidden_layer_type, uint batch_norm, bool _weighted_sample=false,
      uint momentum_=0) :
    size_input_state(input), size_sensors(sensors), size_motors(size_input_state - sensors),
    kMinibatchSize(_kMinibatchSize), weighted_sample(_weighted_sample), hiddens_size(hiddens.size()) {

    ASSERT(alpha > 0, "alpha <= 0");
    ASSERT(_kMinibatchSize > 0, "_kMinibatchSize <= 0");
    ASSERT(hidden_layer_type >= 1 && hidden_layer_type <= 3, "hidden_layer_type not in (1,2,3)");
    batch_norm_type bna = convertBN(batch_norm, hiddens.size());

    if(size_motors == 0 && bna.arch == batch_norm_arch::first_except_action){
      LOG_ERROR("use only value > 15 of batch norm for critic Q");
      exit(-1);
    }

    caffe::SolverParameter solver_param;
    caffe::NetParameter* net_param = solver_param.mutable_net_param();

    caffe::NetParameter net_param_init;
    net_param_init.set_name("Critic");
    net_param_init.set_force_backward(true);
    MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name,"dummy1"},
                    boost::none, {kMinibatchSize, size_sensors, kFixedDim, kFixedDim});
    if(size_motors > 0)
      MemoryDataLayer(net_param_init, action_input_layer_name, {actions_blob_name,"dummy2"},
                      boost::none, {kMinibatchSize, size_motors, kFixedDim, kFixedDim});
    MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy3"},
                    boost::none, {kMinibatchSize, 1, kFixedDim, kFixedDim});
    if(weighted_sample && size_motors > 0) {
      MemoryDataLayer(net_param_init, wsample_input_layer_name, {wsample_blob_name,"dummy4"},
                      boost::none, {kMinibatchSize, 1, kFixedDim, kFixedDim});
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3", "dummy4"}, {}, boost::none);
    } else if(weighted_sample) {
      MemoryDataLayer(net_param_init, wsample_input_layer_name, {wsample_blob_name,"dummy4"},
                      boost::none, {kMinibatchSize, 1, kFixedDim, kFixedDim});
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy3", "dummy4"}, {}, boost::none);
    } else if(size_motors > 0)
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3"}, {}, boost::none);
    else
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy3"}, {}, boost::none);

    std::string tower_top;
    if(bna.arch == batch_norm_arch::first_except_action){
      BatchNormTower(net_param_init, 0, {states_blob_name}, {states_blob_name}, boost::none, bna);
      bna.arch = batch_norm_arch::none;
      ConcatLayer(net_param_init, "concat", {states_blob_name,actions_blob_name}, {states_actions_blob_name}, boost::none, 1);
      tower_top = Tower(net_param_init, states_actions_blob_name, hiddens, bna, hidden_layer_type);
    } else {
      if(size_motors > 0)
        ConcatLayer(net_param_init, "concat", {states_blob_name,actions_blob_name}, {states_actions_blob_name}, boost::none, 1);
      else
        ConcatLayer(net_param_init, "concat", {states_blob_name}, {states_actions_blob_name}, boost::none, 1);
      tower_top = Tower(net_param_init, states_actions_blob_name, hiddens, bna, hidden_layer_type);
      BatchNormTower(net_param_init, hiddens.size(), {tower_top}, {tower_top}, boost::none, bna);
    }
    IPLayer(net_param_init, produce_name("ip", hiddens.size()+1), {tower_top}, {q_values_blob_name}, boost::none, 1);
    if(!weighted_sample)
      EuclideanLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name},
    {loss_blob_name}, boost::none);
    else {
      EuclideanWSLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name, wsample_blob_name},
      {loss_blob_name}, boost::none);
    }

//     net_param_init.PrintDebugString();
    net_param->CopyFrom(net_param_init);

    solver_param.set_type("Adam");
    solver_param.set_max_iter(10000000);
    solver_param.set_base_lr(alpha);
    solver_param.set_lr_policy("fixed");
    solver_param.set_snapshot_prefix("critic");
    if(momentum_ == 0){
      solver_param.set_momentum(0.);
      solver_param.set_momentum2(0.999);
    } else if(momentum_ == 1){
      // same as DDPG of OpenAI Baseline
      solver_param.set_momentum(0.9);
      solver_param.set_momentum2(0.999);
    } else if(momentum_ == 2){
      solver_param.set_momentum(0.);
      solver_param.set_momentum2(0.);      
    } else {
      solver_param.set_momentum(0.9);
      solver_param.set_momentum2(0.);   
    }
    if(!(decay_v < 0)) //-1
      solver_param.set_weight_decay(decay_v);
//     solver_param.set_clip_gradients(10);

    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();
    writeNN_struct(*net_param);
    
//       LOG_DEBUG("param critic : " <<  neural_net->params().size());
    ASSERT(add_loss_layer == false, "only for actor, check please");
  }

  inline std::string produce_name(const std::string& type, uint rank=0, uint task=0) {
    return type + std::to_string(rank) + "_layer." + task_name + std::to_string(task);
  }

//
// ACTOR POLICY
//
  MLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha, uint _kMinibatchSize,
      uint hidden_layer_type, uint last_layer_type, uint batch_norm, bool loss_layer=false, uint momentum_=0) : 
      size_input_state(sensors), size_sensors(sensors), size_motors(motors), kMinibatchSize(_kMinibatchSize), 
      add_loss_layer(loss_layer), hiddens_size(hiddens.size()) {

    ASSERT(alpha > 0, "alpha <= 0");
    ASSERT(_kMinibatchSize > 0, "_kMinibatchSize <= 0");
    ASSERT(hidden_layer_type >= 1 && hidden_layer_type <= 3, "hidden_layer_type not in (1,2,3)");
    if(last_layer_type == 1 || last_layer_type == 3)
    	LOG_INFO("warning : relu as last actor network is not recommanded");
    batch_norm_type bna = convertBN(batch_norm, hiddens.size());

    caffe::SolverParameter solver_param;
    caffe::NetParameter* net_param = solver_param.mutable_net_param();

    caffe::NetParameter net_param_init;
    net_param_init.set_name("Actor");
    net_param_init.set_force_backward(true);

    MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name, "dummy1"},
        boost::none, {kMinibatchSize, size_sensors, kFixedDim, kFixedDim});
    if(!loss_layer)
      SilenceLayer(net_param_init, "silence", {"dummy1"}, {}, boost::none);
    std::string tower_top = Tower(net_param_init,
                                  states_blob_name,
                                  hiddens, bna, hidden_layer_type);
    BatchNormTower(net_param_init, hiddens.size(), {tower_top}, {tower_top}, boost::none, bna);

    std::string layer_name = produce_name("ip", hiddens.size()+1);
    if(last_layer_type == 0)
      IPLayer(net_param_init, layer_name, {tower_top}, {actions_blob_name}, boost::none, motors);
    else if(last_layer_type == 1) {
      IPLayer(net_param_init, layer_name, {tower_top}, {actions_blob_name}, boost::none, motors);
      LReluLayer(net_param_init, produce_name("func", hiddens.size()+1), {actions_blob_name}, {actions_blob_name},
                boost::none);
    } else if(last_layer_type == 2) {
      IPLayer(net_param_init, layer_name, {tower_top}, {actions_blob_name}, boost::none, motors);
      TanhLayer(net_param_init, produce_name("func", hiddens.size()+1), {actions_blob_name}, {actions_blob_name},
                boost::none);
    } else if(last_layer_type == 3) {
      IPLayer(net_param_init, layer_name, {tower_top}, {actions_blob_name}, boost::none, motors);
      ReluLayer(net_param_init, produce_name("func", hiddens.size()+1), {actions_blob_name}, {actions_blob_name},
                boost::none);
    }
    if(bna.arch == batch_norm_arch::very_all){
      BatchNormTower(net_param_init, hiddens.size()+1, {actions_blob_name}, {actions_blob_name}, boost::none, bna);
    }
    if(loss_layer) {
      MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy2"},
                      boost::none, {kMinibatchSize, size_motors, kFixedDim, kFixedDim});
      SilenceLayer(net_param_init, "silence", {"dummy1", "dummy2"}, {}, boost::none);
      EuclideanLossLayer(net_param_init, "loss", {actions_blob_name, targets_blob_name},
      {loss_blob_name}, boost::none);
    }

//     net_param_init.PrintDebugString();
    net_param->CopyFrom(net_param_init);

    solver_param.set_type("Adam");
    solver_param.set_max_iter(10000000);
    solver_param.set_base_lr(alpha);
    solver_param.set_lr_policy("fixed");
    solver_param.set_snapshot_prefix("actor");
    if(momentum_ == 0){
      solver_param.set_momentum(0.);
      solver_param.set_momentum2(0.999);
    } else if(momentum_ == 1){      
      // same as DDPG of OpenAI Baseline 
      solver_param.set_momentum(0.9);
      solver_param.set_momentum2(0.999);
    } else if(momentum_ == 2){
      solver_param.set_momentum(0.);
      solver_param.set_momentum2(0.);      
    } else {
      solver_param.set_momentum(0.9);
      solver_param.set_momentum2(0.);   
    }
//     solver_param.set_clip_gradients(10);//not used by ADAM
//     solver_param.set_display(true);

    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();
    writeNN_struct(*net_param);

//       LOG_DEBUG("actor critic : " <<  neural_net->params().size());
  }

//   
// COPY CONST
// 
  MLP(const MLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) : size_input_state(
      m.size_input_state), size_sensors(m.size_sensors),
    size_motors(m.size_motors), kMinibatchSize(m.kMinibatchSize), add_loss_layer(m.add_loss_layer),
    weighted_sample(m.weighted_sample), hiddens_size(m.hiddens_size) {
    if(!copy_solver) {
      if(_phase == ::caffe::Phase::TRAIN)
        LOG_WARNING("You are copying a net in training phase without solver." << std::endl <<
        "So you BN layer will still learn");

      caffe::NetParameter net_param;
      m.neural_net->ToProto(&net_param);
      net_param.set_force_backward(true);
      net_param.mutable_state()->set_phase(_phase);
      for(int i =0; i < net_param.layer_size(); i++) {
        if(net_param.layer(i).has_batch_norm_param()) {
          net_param.mutable_layer(i)->mutable_batch_norm_param()->set_use_global_stats(_phase == ::caffe::Phase::TEST);
        }
      }
      UpgradeNetBatchNorm(&net_param);
#ifndef NDEBUG
      if(MyNetNeedsUpgrade(net_param)) {
        LOG_DEBUG("network need update");
        exit(1);
      }
#endif
      
//       net_param.PrintDebugString();
      neural_net.reset(new caffe::Net<double>(net_param));
      solver = nullptr;
    } else {
      caffe::SolverParameter solver_param(m.solver->param());
      m.neural_net->ToProto(solver_param.mutable_net_param());
      caffe::NetParameter* net_param = solver_param.mutable_net_param();
      net_param->mutable_state()->set_phase(_phase);
      net_param->set_force_backward(true);
      UpgradeNetBatchNorm(net_param);
      for(int i =0; i < net_param->layer_size(); i++) {
        if(net_param->layer(i).has_batch_norm_param()) {
//           net_param->mutable_layer(i)->clear_param();
          if(_phase == ::caffe::Phase::TRAIN)
            net_param->mutable_layer(i)->mutable_batch_norm_param()->set_use_global_stats(false);
          else
            net_param->mutable_layer(i)->mutable_batch_norm_param()->set_use_global_stats(true);
        }
      }
      
      solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
      neural_net = solver->net();
#ifndef NDEBUG
      if(MyNetNeedsUpgrade(*net_param)) {
        LOG_ERROR("network need update");
        exit(1);
      }
#endif
    }

    if((uint)neural_net->blob_by_name(MLP::states_blob_name)->num() != kMinibatchSize)
      increase_batchsize(kMinibatchSize);
    
    if(number_of_parameters() != m.number_of_parameters()){
      LOG_ERROR("number of param differs");
      LOG_DEBUG(number_of_parameters() << " " << m.number_of_parameters());
      ASSERT(number_of_parameters() == m.number_of_parameters(), "gettrace");
      exit(1);
    }
  }

 private:
  MLP(const MLP&) {
    LOG_ERROR("should not be called");
    exit(1);
  }

 protected:
  MLP(unsigned int _size_input_state, uint _size_sensors, unsigned int _motors, uint _kMinibatchSize, bool loss_layer,
      bool _weighted_sample, uint _hiddens_size) :
    size_input_state(_size_input_state), size_sensors(_size_sensors), size_motors(_motors),
    kMinibatchSize(_kMinibatchSize), add_loss_layer(loss_layer), weighted_sample(_weighted_sample), hiddens_size(_hiddens_size) {
  }

 public:

  virtual ~MLP() {
    if(solver != nullptr)
      delete solver;
  }

  virtual void exploit(boost::property_tree::ptree*, MLP*) {
    LOG_ERROR("should not be called");
    exit(1);
  }
//   void randomizeWeights(const std::vector<uint>& hiddens){
//     //TODO: better
//     caffe::NetParameter net_param_init;
//     net_param_init.set_name("Critic");
//     net_param_init.set_force_backward(true);
//     MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name,"dummy1"},
//                 boost::none, {kMinibatchSize, kFixedDim, size_sensors, 1});
//     MemoryDataLayer(net_param_init, action_input_layer_name, {actions_blob_name,"dummy2"},
//                 boost::none, {kMinibatchSize, kFixedDim, size_motors, 1});
//     MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy3"},
//                 boost::none, {kMinibatchSize, 1, 1, 1});
//     SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3"}, {}, boost::none);
//     ConcatLayer(net_param_init, "concat", {states_blob_name,actions_blob_name}, {states_actions_blob_name}, boost::none, 2);
//     std::string tower_top = Tower(net_param_init, "", states_actions_blob_name, hiddens);
//     IPLayer(net_param_init, q_values_layer_name, {tower_top}, {q_values_blob_name}, boost::none, 1);
//     EuclideanLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name},
//                       {loss_blob_name}, boost::none);
//
//     caffe::NetParameter* net_param = solver_param.mutable_net_param();
//     neural_net.reset(new caffe::Net<double>(net_param));
// //     neural_net->Init(net_param_init);
//
//   }
  
  int get_batchsize() {
        return kMinibatchSize;
  }
  
  void increase_batchsize(uint new_batch_size) {
    kMinibatchSize = new_batch_size;

    neural_net->blob_by_name(MLP::states_blob_name)->Reshape(kMinibatchSize, size_sensors, kFixedDim, kFixedDim);
    auto state_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                               state_input_layer_name));
    state_input_layer->set_batch_size(kMinibatchSize);

    if(size_input_state != size_sensors || add_loss_layer
        || size_motors == 0) { //critic net constructor 1 or actor with loss
      if(size_motors > 0 && !add_loss_layer) { //only for critic with action in inputs
        neural_net->blob_by_name(MLP::actions_blob_name)->Reshape(kMinibatchSize, size_motors, kFixedDim, kFixedDim);
        auto action_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                                    action_input_layer_name));
        action_input_layer->set_batch_size(kMinibatchSize);
      }

      neural_net->blob_by_name(MLP::targets_blob_name)->Reshape(kMinibatchSize, !add_loss_layer ? 1 : size_motors, 
                                                                kFixedDim, kFixedDim);
      auto target_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                                  target_input_layer_name));
      target_input_layer->set_batch_size(kMinibatchSize);

      if(weighted_sample) {
        neural_net->blob_by_name(MLP::wsample_blob_name)->Reshape(kMinibatchSize, 1, kFixedDim, kFixedDim);
        auto wsample_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                                     wsample_input_layer_name));
        wsample_input_layer->set_batch_size(kMinibatchSize);
      }
    }

    neural_net->Reshape();
  }

  virtual void soft_update(const MLP& from, double tau) {
    auto net_from = from.neural_net;
    auto net_to = neural_net;
    // TODO: Test if learnable_params() is sufficient for soft update
//     learnable param already contains batch_norm parameters
    const auto& from_params = net_from->params();
    const auto& to_params = net_to->params();
    CHECK_EQ(from_params.size(), to_params.size());

    for (uint i = 0; i < from_params.size(); ++i) {
      auto& from_blob = from_params[i];
      auto& to_blob = to_params[i];
      caffe::caffe_cpu_axpby(from_blob->count(), tau, from_blob->cpu_data(),
                             (1.f-tau), to_blob->mutable_cpu_data());
    }
  }

  void learn(const std::vector<double>& sensors, const std::vector<double>& motors) {
    InputDataIntoLayers(&sensors, nullptr, &motors);
    solver->Step(1);
  }

  void learn(const std::vector<double>& sensors, const std::vector<double>& motors, double q) {
    std::vector<double> target({q});
    if(size_motors > 0)
      InputDataIntoLayers(&sensors, &motors, &target);
    else
      InputDataIntoLayers(&sensors, nullptr, &target);
    solver->Step(1);
  }

  void learn_batch(const std::vector<double>& sensors, const std::vector<double>& motors, const std::vector<double>& q, uint iter) {
    if(size_motors > 0 && !add_loss_layer)
      InputDataIntoLayers(&sensors, &motors, &q);
    else
      InputDataIntoLayers(&sensors, nullptr, &q);
    if(!ewc_enabled())
      solver->Step(iter);
    else {
      for(uint i=0;i<iter;i++){
        neural_net->ClearParamDiffs();
        neural_net->ForwardBackward();
        updateFisher(kMinibatchSize);
        regularize();
        solver->ApplyUpdate();
        solver->set_iter(solver->iter() + 1);
      }
    }
  }

  void learn_batch_lw(const std::vector<double>& sensors, const std::vector<double>& motors, const std::vector<double>& q,
                      const std::vector<double>& lw, uint iter) {
    if(size_motors > 0)
      InputDataIntoLayers(&sensors, &motors, &q);
    else 
      InputDataIntoLayers(&sensors, nullptr, &q);
    setWeightedSampleVector(&lw, false);
    if(!ewc_enabled())
      solver->Step(iter);
    else {
      for(uint i=0;i<iter;i++){
        neural_net->ClearParamDiffs();
        neural_net->ForwardBackward();
        updateFisher(kMinibatchSize);
        regularize();
        solver->ApplyUpdate();
        solver->set_iter(solver->iter() + 1);
      }
    }
  }

  virtual double computeOutVF(const std::vector<double>& sensors, const std::vector<double>& motors) {
    std::vector<double> states_input(size_sensors * kMinibatchSize, 0.0f);
    std::copy(sensors.begin(), sensors.end(), states_input.begin());

    if(weighted_sample)
      setWeightedSampleVector(nullptr, true);

    if(size_motors > 0) {
      std::vector<double> actions_input(size_motors * kMinibatchSize, 0.0f);
      std::copy(motors.begin(), motors.end(), actions_input.begin());

      InputDataIntoLayers(&states_input, &actions_input, nullptr, true);
      neural_net->Forward(nullptr); //actions_input will be erase so let me here
    } else {
      InputDataIntoLayers(&states_input, nullptr, nullptr, true);
      neural_net->Forward(nullptr);
    }

    const auto q_values_blob = neural_net->blob_by_name(q_values_blob_name);

    return q_values_blob->data_at(0, 0, 0, 0);
  }

  virtual std::vector<double>* computeOutVFBatch(const std::vector<double>& sensors, const std::vector<double>& motors) {
    if(size_motors > 0)
      InputDataIntoLayers(&sensors, &motors, nullptr, true);
    else 
      InputDataIntoLayers(&sensors, nullptr, nullptr, true);
    
    if(weighted_sample)
      setWeightedSampleVector(nullptr, true);
    neural_net->Forward(nullptr);

    const auto q_values_blob = neural_net->blob_by_name(q_values_blob_name);
    auto outputs = new std::vector<double>(kMinibatchSize);

    uint i=0;
    for (uint n = 0; n < kMinibatchSize; ++n)
      outputs->at(i++) = q_values_blob->data_at(n, 0, 0, 0);

    return outputs;
  }

  virtual std::vector<double>* computeOut(const std::vector<double>& states_batch) {
    std::vector<double> states_input(size_input_state * kMinibatchSize, 0.0f);
    std::copy(states_batch.begin(), states_batch.end(),states_input.begin());

    InputDataIntoLayers(&states_input, NULL, NULL, add_loss_layer);
    neural_net->Forward(nullptr);

    std::vector<double>* outputs = new std::vector<double>(size_motors);
    const auto actions_blob = neural_net->blob_by_name(actions_blob_name);

    for(uint j=0; j < outputs->size(); j++)
      outputs->at(j) = actions_blob->data_at(0, j, 0, 0);

    return outputs;
  }

  std::vector<double>* computeOutBatch(const std::vector<double>& in) {
    InputDataIntoLayers(&in, NULL, NULL, add_loss_layer);
    neural_net->Forward(nullptr);

    auto outputs = new std::vector<double>(kMinibatchSize * size_motors);
    const auto actions_blob = neural_net->blob_by_name(actions_blob_name);
    uint i=0;
    for (uint n = 0; n < kMinibatchSize; ++n)
      for(uint j=0; j < size_motors; j++)
        outputs->at(i++) = actions_blob->data_at(n, j, 0, 0);

    return outputs;
  }

  double weight_l1_norm(bool ignore_bn_weight=false) {
    (void) ignore_bn_weight;
    double sum = 0.f;

    caffe::Net<double>& net = *neural_net;
    const double* weights;

    for (uint i = 0; i < net.learnable_params().size(); ++i) {
      auto blob = net.learnable_params()[i];
#ifdef CAFFE_CPU_ONLY
      weights = blob->cpu_data();
      sum += caffe::caffe_cpu_asum(blob->count(), weights);
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        weights = blob->cpu_data();
        sum += caffe::caffe_cpu_asum(blob->count(), weights);
        break;
      case caffe::Caffe::GPU:
        weights = blob->gpu_data();
        double c=0;
        caffe::caffe_gpu_asum(blob->count(), weights, &c);
        sum += c;
        break;
      }
#endif
    }

    return sum;
  }
  
  uint number_of_parameters(bool ignore_null_lr=false) const {
    uint n = 0;
    caffe::Net<double>& net = *neural_net;
    ASSERT(net.learnable_params().size() == net.params_lr().size(), "failed");
    for (uint i = 0; i < net.learnable_params().size(); ++i) 
      if(!ignore_null_lr || net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        n += blob->count();
      }

    return n;
  }

  void copyWeightsTo(double* startx, bool ignore_null_lr) const {
    uint index = 0;
    
    caffe::Net<double>& net = *neural_net;
    const double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) {
      if(!ignore_null_lr || net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        weights = blob->cpu_data();
        for(int n=0; n < blob->count(); n++) {
          startx[index] = weights[n];
          index++;
        }
      }
    }
  }
  
  void copyDiffTo(double* startx, bool ignore_null_lr) const {
    uint index = 0;
    
    caffe::Net<double>& net = *neural_net;
    const double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) {
      if(!ignore_null_lr || net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        weights = blob->cpu_data();
        for(int n=0; n < blob->count(); n++) {
          startx[index] = weights[n];
          index++;
        }
      }
    }
  }

  virtual void copyWeightsFrom(const double* startx, bool ignore_null_lr) {
    uint index = 0;

    caffe::Net<double>& net = *neural_net;
    double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) 
      if(!ignore_null_lr || net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        weights = blob->mutable_cpu_data();
        for(int n=0; n < blob->count(); n++) {
          weights[n] = startx[index];
          index++;
        }
      }
  }
  
  virtual void copyDiffFrom(const double* startx, bool ignore_null_lr) {
    uint index = 0;

    caffe::Net<double>& net = *neural_net;
    double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) 
      if(!ignore_null_lr || net.params_lr()[i] != 0.0f) {
        auto blob = net.learnable_params()[i];
        weights = blob->mutable_cpu_diff();

        for(int n=0; n < blob->count(); n++) {
          weights[n] = startx[index];
          index++;
        }
      }
  }

  double error() {
    auto blob = neural_net->blob_by_name(MLP::loss_blob_name);
    double sum = 0.f;
    const double* errors;
#ifdef CAFFE_CPU_ONLY
    errors = blob->cpu_data();
    sum = caffe::caffe_cpu_asum(blob->count(), errors);
#else
    switch (caffe::Caffe::mode()) {
        case caffe::Caffe::CPU:
          errors = blob->cpu_data();
          sum = caffe::caffe_cpu_asum(blob->count(), errors);
          break;
        case caffe::Caffe::GPU:
          errors = blob->gpu_data();
          caffe::caffe_gpu_asum(blob->count(), errors, &sum);
          break;
    }
#endif

    return sum;
  }
  
  virtual double ewc_cost() {
    return 0.f;
  }
  
  virtual bool ewc_enabled() {
    return false;
  }
    
  virtual void update_best_param_previous_task(double) {
    
  }
  
  virtual void regularize() {
    
  }
  
  virtual void ewc_decay_update(){
    
  }
  
  virtual void updateFisher(double) {
    
  }
  
  virtual void neutral_action(const std::vector<double>&, std::vector<double>*){
    
  }
  
  virtual uint ewc_best_method() {
    return 0;
  }
  
  bool isCritic(){
    return size_input_state != size_sensors || size_motors == 0;
  }

  boost::shared_ptr<caffe::Net<double>> getNN() {
    return neural_net;
  }

  caffe::Solver<double>* getSolver() {
    return solver;
  }
  
  void copyParametersFrom(const MLP * from){
    neural_net->ShareTrainedLayersWith(from->neural_net.get());
  }

protected:
  std::string Tower(caffe::NetParameter& np,
                    const std::string& input_blob_name,
                    const std::vector<uint>& layer_sizes,
                    const batch_norm_type& bna,
                    uint hidden_layer_type,
                    uint link_struct=0,
                    uint task=0,
                    bool avoid_states_input=false, 
                    bool policy=true) {
    std::string input_name = input_blob_name;
    std::string layer_name;

    for (uint i=1; i<layer_sizes.size()+1; ++i) {
      if(i != 1 || !avoid_states_input || !policy)
        BatchNormTower(np, i-1, {input_name}, {input_name}, boost::none, bna, task);

      if(link_struct != 0) {
        std::vector<std::string> bottoms;
        bottoms.push_back(input_name);

        if(link_struct & (1 << 0)) {
          if(i==1) {
            if(!avoid_states_input){
              if(policy)
                bottoms.push_back(states_blob_name+"."+task_name+std::to_string(task-1));
              else
                bottoms.push_back(states_actions_blob_name+"."+task_name+std::to_string(task-1));
            }
          } else //if(bottoms.size() == 1)//to be precomputed
            bottoms.push_back(produce_name("ip", i - 1, task -1));
        }

        if(link_struct & (1 << 1)) {
          if(i==1)
            bottoms.push_back(produce_name("rsh", i, task -1));
          else
            bottoms.push_back(produce_name("ip", i, task -1));
        }

        if(link_struct & (1 << 2)) {
          if(i==1)
            bottoms.push_back(produce_name("rsh", i+1, task -1));
          else
            bottoms.push_back(produce_name("ip", i+1, task -1));
        }
        layer_name = produce_name("cc", i, task);
        ConcatLayer(np, layer_name, bottoms, {layer_name}, boost::none, 1);
        input_name = layer_name;
      }

      layer_name = produce_name("ip", i, task);
      std::string top_name = produce_name("ip", i, task);
      IPLayer(np, layer_name, {input_name}, {top_name}, boost::none, layer_sizes[i-1]);
      layer_name = produce_name("func", i, task);
      if(hidden_layer_type==1)
        LReluLayer(np, layer_name, {top_name}, {top_name}, boost::none);
      else if(hidden_layer_type==2)
        TanhLayer(np, layer_name, {top_name}, {top_name}, boost::none);
      else if(hidden_layer_type==3)
        ReluLayer(np, layer_name, {top_name}, {top_name}, boost::none);
      else {
        LOG_ERROR("hidden_layer_type " << hidden_layer_type << "not implemented");
        exit(1);
      }
      input_name = top_name;
    }
    return input_name;
  }
  
  int get_layer_index(const std::string& layer_name) {
    caffe::Net<double>& net = *neural_net;
    if (!net.has_layer(layer_name)) {
      return -1;
    }
    const std::vector<std::string>& layer_names = net.layer_names();
    int indx = std::distance(
      layer_names.begin(),
                             std::find(layer_names.begin(), layer_names.end(), layer_name));
    return indx;
  }

  //Network config methods, directly copied from dqn.cpp
  void PopulateLayer(caffe::LayerParameter& layer,
                     const std::string& name, const std::string& type,
                     const std::vector<std::string>& bottoms,
                     const std::vector<std::string>& tops,
                     const boost::optional<caffe::Phase>& include_phase) {
    layer.set_name(name);
    layer.set_type(type);
    for (auto& bottom : bottoms) {
      layer.add_bottom(bottom);
    }
    for (auto& top : tops) {
      layer.add_top(top);
    }
    if (include_phase) {
      layer.add_include()->set_phase(*include_phase);
    }
  }
  void ReshapeLayer(caffe::NetParameter& net_param,
                    const std::string& name,
                    const std::vector<std::string>& bottoms,
                    const std::vector<std::string>& tops,
                    const boost::optional<caffe::Phase>& include_phase,
                    const std::vector<uint>& _shape) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Reshape", bottoms, tops, include_phase);
    caffe::ReshapeParameter* reshape_param = layer.mutable_reshape_param();
    for(auto i : _shape)
      reshape_param->mutable_shape()->add_dim(i);
  }
  void ConcatLayer(caffe::NetParameter& net_param,
                   const std::string& name,
                   const std::vector<std::string>& bottoms,
                   const std::vector<std::string>& tops,
                   const boost::optional<caffe::Phase>& include_phase,
                   const int& axis) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Concat", bottoms, tops, include_phase);
    caffe::ConcatParameter* concat_param = layer.mutable_concat_param();
    concat_param->set_axis(axis);
  }
  void MemoryDataLayer(caffe::NetParameter& net_param,
                       const std::string& name,
                       const std::vector<std::string>& tops,
                       const boost::optional<caffe::Phase>& include_phase,
                       const std::vector<uint>& shape) {
    caffe::LayerParameter& memory_layer = *net_param.add_layer();
    PopulateLayer(memory_layer, name, "MemoryData", {}, tops, include_phase);
    CHECK_EQ(shape.size(), 4);
    caffe::MemoryDataParameter* memory_data_param = memory_layer.mutable_memory_data_param();
    memory_data_param->set_batch_size(shape[0]);
    memory_data_param->set_channels(shape[1]);
    memory_data_param->set_height(shape[2]);
    memory_data_param->set_width(shape[3]);
  }
  void SilenceLayer(caffe::NetParameter& net_param,
                    const std::string& name,
                    const std::vector<std::string>& bottoms,
                    const std::vector<std::string>& tops,
                    const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Silence", bottoms, tops, include_phase);
  }
  void LReluLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
    caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
    relu_param->set_negative_slope(0.01);
  }
  void ReluLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
  }
  void SliceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase, int until) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
    caffe::SliceParameter* slice_param = layer.mutable_slice_param();
//     slice_param->set
    slice_param->set_axis(1);
    slice_param->add_slice_point(until);
  }
  void TanhLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "TanH", bottoms, tops, include_phase);
  }
  void IPLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase,
               const int num_output) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "InnerProduct", bottoms, tops, include_phase);
    caffe::InnerProductParameter* ip_param = layer.mutable_inner_product_param();
    ip_param->set_num_output(num_output);
    caffe::FillerParameter* weight_filler = ip_param->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.01);
//     LOG_DEBUG("" << (ip_param->has_bias_term() ? "oui" : "non"));
//    LOG_DEBUG(ip_param->has_bias_term());
//     ip_param->set_bias_term(true);
    caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
    bias_filler->set_type("gaussian");
    bias_filler->set_std(0.01);
//     bias_filler->set_type("constant");
//     bias_filler->set_value(1);
//     bias already here but not in first blob but set to 0?
//     can learn layer_param().inner_product_param().bias_term(); -> true
  }
  void EuclideanLossLayer(caffe::NetParameter& net_param,
                          const std::string& name,
                          const std::vector<std::string>& bottoms,
                          const std::vector<std::string>& tops,
                          const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "EuclideanLoss", bottoms, tops, include_phase);
  }
  void EuclideanWSLossLayer(caffe::NetParameter& net_param,
                            const std::string& name,
                            const std::vector<std::string>& bottoms,
                            const std::vector<std::string>& tops,
                            const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "EuclideanWSLoss", bottoms, tops, include_phase);
  }
  void SoftmaxLayer(caffe::NetParameter& net_param,
                    const std::string& name,
                    const std::vector<std::string>& bottoms,
                    const std::vector<std::string>& tops,
                    const boost::optional<caffe::Phase>& include_phase,
                    const int axis) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Softmax", bottoms, tops, include_phase);
    caffe::SoftmaxParameter* param = layer.mutable_softmax_param();
    param->set_axis(axis);
  }

  void BatchNormTower(caffe::NetParameter& net_param,
                      uint rank,
                      const std::vector<std::string>& bottoms,
                      const std::vector<std::string>& tops,
                      const boost::optional<caffe::Phase>& include_phase,
                      const batch_norm_type& bna,
                      uint task=0) {
    if(bna.arch == batch_norm_arch::none ||
        (bna.arch == batch_norm_arch::first && rank > 0) ||
        (bna.arch == batch_norm_arch::all_except_last && rank >= bna.max_rank ))
      return;

    std::string layer_name = produce_name("bn", rank, task);
    BatchNormLayer(net_param, layer_name, bottoms, tops, include_phase);
    if(bna.with_scale) {
      layer_name = produce_name("sc", rank, task);
      ScaleLayer(net_param, layer_name, tops, tops, include_phase, bna);
    }
  }

  void BatchNormLayer(caffe::NetParameter& net_param,
                      const std::string& name,
                      const std::vector<std::string>& bottoms,
                      const std::vector<std::string>& tops,
                      const boost::optional<caffe::Phase>& include_phase) {

    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "BatchNorm", bottoms, tops, include_phase);
    caffe::BatchNormParameter* param = layer.mutable_batch_norm_param();//let me know I am batchnorm
    param->set_use_global_stats(false); //let me learn first
  }

  void ScaleLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const batch_norm_type& bna
                 ) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    if(bna.with_scale_bias)
      layer.mutable_scale_param()->set_bias_term(true);
    PopulateLayer(layer, name, "Scale", bottoms, tops, include_phase);
  }

  typedef caffe::MemoryDataLayer<double> mdatal;
  //keep me private because I can change my inputs
  void InputDataIntoLayers(const std::vector<double>* states_input, 
                           const std::vector<double>* actions_input, 
                           const std::vector<double>* target_input,
                           bool local_initizialition_target=false) {
    if (states_input != nullptr) {
      copy_states.reset(new std::vector<double>(*states_input));
      auto layer = neural_net->layer_by_name(state_input_layer_name);
      auto state_input_layer = boost::dynamic_pointer_cast<mdatal>(layer);
      CHECK(state_input_layer);
      ASSERT(copy_states->size() == state_input_layer->batch_size() * size_sensors,
             "size pb " << copy_states->size() << " " << state_input_layer->batch_size() * size_sensors);
      state_input_layer->Reset(copy_states->data(), copy_states->data(), state_input_layer->batch_size());
    }
    if (actions_input != nullptr) {
      copy_actions.reset(new std::vector<double>(*actions_input));
      auto layer = neural_net->layer_by_name(action_input_layer_name);
      auto action_input_layer = boost::dynamic_pointer_cast<mdatal>(layer);
      CHECK(action_input_layer);
      ASSERT(copy_actions->size() == action_input_layer->batch_size() * size_motors, "size pb");
      action_input_layer->Reset(copy_actions->data(), copy_actions->data(), action_input_layer->batch_size());
    }
    if (target_input != nullptr || local_initizialition_target) {
      if(!local_initizialition_target)
        copy_q_values.reset(new std::vector<double>(*target_input));
      else if(add_loss_layer) //called by actor if he have a loss
        copy_q_values.reset(new std::vector<double>(kMinibatchSize * size_motors, 0.0f));
      else
        copy_q_values.reset(new std::vector<double>(kMinibatchSize, 0.0f));
      auto layer = neural_net->layer_by_name(target_input_layer_name);
      auto target_input_layer = boost::dynamic_pointer_cast<mdatal>(layer);
      CHECK(target_input_layer);
#ifndef NDEBUG
      if(add_loss_layer) //called by actor if he have a loss or a value function V
        ASSERT((uint)copy_q_values->size() == target_input_layer->batch_size() * size_motors, 
               "size pb " << copy_q_values->size() << " " << target_input_layer->batch_size() * size_motors);
      else
        ASSERT((int)copy_q_values->size() == target_input_layer->batch_size(), "size pb " << copy_q_values->size() << " " << target_input_layer->batch_size());
#endif
      target_input_layer->Reset(copy_q_values->data(), copy_q_values->data(), target_input_layer->batch_size());
    }
  }
  
  bool MyNetNeedsUpgrade(const caffe::NetParameter& net_param) {
    return NetNeedsV0ToV1Upgrade(net_param) || NetNeedsV1ToV2Upgrade(net_param)
    || NetNeedsDataUpgrade(net_param) || NetNeedsInputUpgrade(net_param);
//     || NetNeedsBatchNormUpgrade(net_param); test wrong written by caffe
  }
  
  void setWeightedSampleVector(const std::vector<double> *wsample_input, bool local_initilization) {
    if(!local_initilization)
      copy_label_weights.reset(new std::vector<double>(*wsample_input));
    else
      copy_label_weights.reset(new std::vector<double>(kMinibatchSize, 0.f));
    auto layer = neural_net->layer_by_name(wsample_input_layer_name);
    auto wsample_input_layer = boost::dynamic_pointer_cast<mdatal>(layer);
    CHECK(wsample_input_layer);
    ASSERT((int)copy_label_weights->size() == wsample_input_layer->batch_size(), "size pb");
    wsample_input_layer->Reset(copy_label_weights->data(), copy_label_weights->data(), wsample_input_layer->batch_size());
  }
  
public:
  void ZeroGradParameters() {
    caffe::Net<double>& net = *neural_net;
    for (uint i = 0; i < net.params().size(); ++i) {
      caffe::shared_ptr<caffe::Blob<double> > blob = net.params()[i];
#ifdef CAFFE_CPU_ONLY
      caffe::caffe_set(blob->count(), static_cast<double>(0),
                       blob->mutable_cpu_diff());
#else
      switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_set(blob->count(), static_cast<double>(0),
                         blob->mutable_cpu_diff());
        break;

      case caffe::Caffe::GPU:
        caffe::caffe_gpu_set(blob->count(), static_cast<double>(0),
                             blob->mutable_gpu_diff());
        break;
      }
#endif
    }
  }

  void critic_backward(){
    int layer = get_layer_index(produce_name("ip", hiddens_size+1));
    
    ASSERT(layer != -1 , "failed to found layer for backward");
    
    neural_net->BackwardFrom(layer);
  }
  
  virtual void actor_backward() {
    int layer = get_layer_index(produce_name("func", hiddens_size+1));//try func
    if(layer == -1)//if there is no activation function
      layer = get_layer_index(produce_name("ip", hiddens_size+1));
    
    ASSERT(layer != -1 , "failed to found layer for backward");
    
    neural_net->BackwardFrom(layer);
  }

  void save(const std::string& path) {
    if(solver == nullptr){
      LOG_DEBUG("WARNING, you are saving a net without solver");
      return;
    }
    solver->set_filename(path);
    solver->Snapshot();
  }

  void load(const std::string& path) {
    std::string rpath = path + ".solverstate.data";
    solver->set_filename(path);
    solver->Restore(rpath.c_str());
//     neural_net->CopyTrainedLayersFrom(path);
  }
  
  static void writeNN_struct(const caffe::NetParameter& param, uint task=0){
    std::string path = param.name() + ".struct.data";
    if(task != 0)
      path = param.name() + "." + std::to_string(task) + ".struct.data";
    std::ofstream ofs (path, std::ofstream::out);
    ofs << param.DebugString();
    ofs.close();
  }

 protected:
  caffe::Solver<double>* solver;
  boost::shared_ptr<caffe::Net<double>> neural_net;
  uint size_input_state;
  uint size_sensors;
  uint size_motors;
  uint kMinibatchSize;
  bool add_loss_layer=false;
  bool weighted_sample=false;
  uint hiddens_size;
//   internal copy of inputs/outputs
  boost::shared_ptr<std::vector<double>> copy_states;
  boost::shared_ptr<std::vector<double>> copy_actions;
  boost::shared_ptr<std::vector<double>> copy_q_values;
  boost::shared_ptr<std::vector<double>> copy_label_weights;
};

#endif  // MLP_HPP



