
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

//depends on CAFFE_CPU_ONLY

class MLP {
 public:

  // Layer Names
  constexpr static auto state_input_layer_name         = "state_input_layer";
  constexpr static auto action_input_layer_name        = "action_input_layer";
  constexpr static auto target_input_layer_name        = "target_input_layer";
  constexpr static auto wsample_input_layer_name       = "wsample_input_layer";
  constexpr static auto q_values_layer_name            = "q_values_layer";
  // Blob names
  constexpr static auto states_blob_name        = "states";
  constexpr static auto actions_blob_name       = "actions";
  constexpr static auto targets_blob_name       = "target";
  constexpr static auto wsample_blob_name       = "wsample";
  constexpr static auto q_values_blob_name      = "q_values";
  constexpr static auto loss_blob_name          = "loss";

  constexpr static auto kStateInputCount = 1;

  MLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
      uint _kMinibatchSize, double decay_v, uint batch_norm, bool _weighted_sample=false) :
    size_input_state(input), size_sensors(sensors), size_motors(size_input_state - sensors),
    kMinibatchSize(_kMinibatchSize), weighted_sample(_weighted_sample) {

    caffe::SolverParameter solver_param;
    caffe::NetParameter* net_param = solver_param.mutable_net_param();

    caffe::NetParameter net_param_init;
    net_param_init.set_name("Critic");
    net_param_init.set_force_backward(true);
    MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name,"dummy1"},
                    boost::none, {kMinibatchSize, kStateInputCount, size_sensors, 1});
    MemoryDataLayer(net_param_init, action_input_layer_name, {actions_blob_name,"dummy2"},
                    boost::none, {kMinibatchSize, kStateInputCount, size_motors, 1});
    MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy3"},
                    boost::none, {kMinibatchSize, 1, 1, 1});
    if(weighted_sample){
      MemoryDataLayer(net_param_init, wsample_input_layer_name, {wsample_blob_name,"dummy4"},
                      boost::none, {kMinibatchSize, 1, 1, 1});
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3", "dummy4"}, {}, boost::none);
    } else
      SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3"}, {}, boost::none);
    ConcatLayer(net_param_init, "concat", {states_blob_name,actions_blob_name}, {"state_actions"}, boost::none, 2);
    std::string tower_top = Tower(net_param_init, "", "state_actions", hiddens, batch_norm);
    IPLayer(net_param_init, q_values_layer_name, {tower_top}, {q_values_blob_name}, boost::none, 1);
    if(!weighted_sample)
      EuclideanLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name},
          {loss_blob_name}, boost::none);
    else {
      EuclideanWSLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name, wsample_blob_name},
          {loss_blob_name}, boost::none);
    }

    net_param->CopyFrom(net_param_init);

    solver_param.set_type("Adam");
    solver_param.set_max_iter(10000000);
    solver_param.set_base_lr(alpha);
    solver_param.set_lr_policy("fixed");
//       solver_param.set_momentum(0.95);
//       solver_param.set_momentum2(0.999);
    if(!(decay_v < 0)) //-1
      solver_param.set_weight_decay(decay_v);
    solver_param.set_clip_gradients(10);

    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();

//       LOG_DEBUG("param critic : " <<  neural_net->params().size());
  }

  MLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha, uint _kMinibatchSize,
      uint last_layer_type, uint batch_norm) : size_input_state(sensors),
    size_sensors(sensors), size_motors(motors), kMinibatchSize(_kMinibatchSize) {

    caffe::SolverParameter solver_param;
    caffe::NetParameter* net_param = solver_param.mutable_net_param();

    caffe::NetParameter net_param_init;
    net_param_init.set_name("Actor");
    net_param_init.set_force_backward(true);
    MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name,"dummy1"},
                    boost::none, {kMinibatchSize, kStateInputCount, size_sensors, 1});
    SilenceLayer(net_param_init, "silence", {"dummy1"}, {}, boost::none);
    std::string tower_top = Tower(net_param_init, "", states_blob_name, hiddens, batch_norm);
    if(last_layer_type == 0)
      IPLayer(net_param_init, "action_layer", {tower_top}, {actions_blob_name}, boost::none, motors);
    else if(last_layer_type == 1) {
      IPLayer(net_param_init, "action_layer_ip", {tower_top}, {"last_relu"}, boost::none, motors);
      ReluLayer(net_param_init, "action_layer", {"last_relu"}, {actions_blob_name}, boost::none);
    } else if(last_layer_type == 2) {
      IPLayer(net_param_init, "action_layer_ip", {tower_top}, {"last_relu"}, boost::none, motors);
      ReluLayer(net_param_init, "action_layer", {"last_relu"}, {actions_blob_name}, boost::none);
    }

    net_param->CopyFrom(net_param_init);

    solver_param.set_type("Adam");
    solver_param.set_max_iter(10000000);
    solver_param.set_base_lr(alpha);
    solver_param.set_lr_policy("fixed");
//       solver_param.set_momentum(0.95);
//       solver_param.set_momentum2(0.999);
    solver_param.set_clip_gradients(10);

    solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
    neural_net = solver->net();

//       LOG_DEBUG("actor critic : " <<  neural_net->params().size());
  }

  MLP(const MLP& m, bool copy_solver) : size_input_state(m.size_input_state), size_sensors(m.size_sensors),
    size_motors(m.size_motors), kMinibatchSize(m.kMinibatchSize), weighted_sample(m.weighted_sample) {
    if(!copy_solver) {
      caffe::NetParameter net_param;
      m.neural_net->ToProto(&net_param);
      net_param.set_force_backward(true);

      neural_net.reset(new caffe::Net<double>(net_param));
      solver = nullptr;
    } else {
      caffe::SolverParameter solver_param(m.solver->param());
      m.neural_net->ToProto(solver_param.mutable_net_param());
      solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
      neural_net = solver->net();
    }
  }

  MLP(const MLP& m, bool _add_loss_layer, bool copy_have_solver) :
    size_input_state(m.size_input_state), size_sensors(m.size_sensors), size_motors(m.size_motors),
    kMinibatchSize(m.kMinibatchSize), add_loss_layer(_add_loss_layer), weighted_sample(m.weighted_sample) {
    caffe::NetParameter net_param;
    m.neural_net->ToProto(&net_param);
    net_param.set_force_backward(true);
    if(add_loss_layer) {
      MemoryDataLayer(net_param, target_input_layer_name, {targets_blob_name,"dummy2"},
                      boost::none, {kMinibatchSize, 1, size_motors, 1});
      EuclideanLossLayer(net_param, "loss", {actions_blob_name, targets_blob_name},
      {loss_blob_name}, boost::none);
    }

    //neural_net.reset(new caffe::Net<double>(net_param));
    if(copy_have_solver) {
      caffe::SolverParameter solver_param(m.solver->param());
      solver_param.mutable_net_param()->CopyFrom(net_param);
      solver = caffe::SolverRegistry<double>::CreateSolver(solver_param);
      neural_net = solver->net();
    } else
      LOG_ERROR("to be implemented");
  }

 private:
  MLP(const MLP&) {
    LOG_ERROR("should not be called");
  }
 public:

  virtual ~MLP() {
    if(solver != nullptr)
      delete solver;
  }


//   void randomizeWeights(const std::vector<uint>& hiddens){
//     //TODO: better
//     caffe::NetParameter net_param_init;
//     net_param_init.set_name("Critic");
//     net_param_init.set_force_backward(true);
//     MemoryDataLayer(net_param_init, state_input_layer_name, {states_blob_name,"dummy1"},
//                 boost::none, {kMinibatchSize, kStateInputCount, size_sensors, 1});
//     MemoryDataLayer(net_param_init, action_input_layer_name, {actions_blob_name,"dummy2"},
//                 boost::none, {kMinibatchSize, kStateInputCount, size_motors, 1});
//     MemoryDataLayer(net_param_init, target_input_layer_name, {targets_blob_name,"dummy3"},
//                 boost::none, {kMinibatchSize, 1, 1, 1});
//     SilenceLayer(net_param_init, "silence", {"dummy1","dummy2","dummy3"}, {}, boost::none);
//     ConcatLayer(net_param_init, "concat", {states_blob_name,actions_blob_name}, {"state_actions"}, boost::none, 2);
//     std::string tower_top = Tower(net_param_init, "", "state_actions", hiddens);
//     IPLayer(net_param_init, q_values_layer_name, {tower_top}, {q_values_blob_name}, boost::none, 1);
//     EuclideanLossLayer(net_param_init, "loss", {q_values_blob_name, targets_blob_name},
//                       {loss_blob_name}, boost::none);
//
//     caffe::NetParameter* net_param = solver_param.mutable_net_param();
//     neural_net.reset(new caffe::Net<double>(net_param));
// //     neural_net->Init(net_param_init);
//
//   }

  void increase_batchsize(uint new_batch_size) {
    kMinibatchSize = new_batch_size;

    neural_net->blob_by_name(MLP::states_blob_name)->Reshape(kMinibatchSize, kStateInputCount, size_sensors, 1);
    auto state_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                               state_input_layer_name));
    state_input_layer->set_batch_size(kMinibatchSize);

    if(size_input_state != size_sensors) {
      neural_net->blob_by_name(MLP::actions_blob_name)->Reshape(kMinibatchSize, kStateInputCount, size_motors, 1);
      auto action_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                                  action_input_layer_name));
      action_input_layer->set_batch_size(kMinibatchSize);

      neural_net->blob_by_name(MLP::targets_blob_name)->Reshape(kMinibatchSize, kStateInputCount, 1, 1);
      auto target_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
                                  target_input_layer_name));
      target_input_layer->set_batch_size(kMinibatchSize);
      
      if(weighted_sample){
        neural_net->blob_by_name(MLP::wsample_blob_name)->Reshape(kMinibatchSize, kStateInputCount, 1, 1);
        auto wsample_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(neural_net->layer_by_name(
          wsample_input_layer_name));
        wsample_input_layer->set_batch_size(kMinibatchSize);
      }
    }

    neural_net->Reshape();
  }

  void soft_update(const MLP& from, double tau) {
    auto net_from = from.neural_net;
    auto net_to = neural_net;
    // TODO: Test if learnable_params() is sufficient for soft update
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

  double computeOutVF(const std::vector<double>& sensors, const std::vector<double>& motors) {
    std::vector<double> states_input(size_sensors * kMinibatchSize, 0.0f);
    std::copy(sensors.begin(), sensors.end(),states_input.begin());

    std::vector<double> actions_input(size_motors * kMinibatchSize, 0.0f);
    std::copy(motors.begin(), motors.end(),actions_input.begin());

    std::vector<double> target_input(kMinibatchSize, 0.0f);

    InputDataIntoLayers(states_input.data(), actions_input.data(), target_input.data());
    neural_net->Forward(nullptr);

    const auto actions_blob = neural_net->blob_by_name(actions_blob_name);

    return actions_blob->data_at(0, 0, 0, 0);
  }

  std::vector<double>* computeOutVFBatch(std::vector<double>& sensors, std::vector<double>& motors) {
    std::vector<double> target_input(kMinibatchSize, 0.0f);
    std::vector<double> target_input2(kMinibatchSize, 0.0f);
    
    InputDataIntoLayers(sensors.data(), motors.data(), target_input.data());
    if(weighted_sample)
      setWeightedSampleVector(target_input2.data());
    neural_net->Forward(nullptr);

    const auto q_values_blob = neural_net->blob_by_name(q_values_blob_name);
    auto outputs = new std::vector<double>(kMinibatchSize);

    uint i=0;
    for (uint n = 0; n < kMinibatchSize; ++n)
      outputs->at(i++) = q_values_blob->data_at(n, 0, 0, 0);

    return outputs;
  }

  std::vector<double>* computeOut(const std::vector<double>& states_batch) {
    std::vector<double> states_input(size_input_state * kMinibatchSize, 0.0f);
    std::copy(states_batch.begin(), states_batch.end(),states_input.begin());
    std::vector<double>* target_input = nullptr;

    if(add_loss_layer) {
      target_input = new std::vector<double>(size_motors * kMinibatchSize, 0.0f);
      InputDataIntoLayers(states_input.data(), NULL, target_input->data());
    } else
      InputDataIntoLayers(states_input.data(), NULL, NULL);
    neural_net->Forward(nullptr);

    if(add_loss_layer)
      delete target_input;

    std::vector<double>* outputs = new std::vector<double>(size_motors);
    const auto actions_blob = neural_net->blob_by_name(actions_blob_name);

    for(uint j=0; j < outputs->size(); j++)
      outputs->at(j) = actions_blob->data_at(0, j, 0, 0);

    return outputs;
  }

  std::vector<double>* computeOutBatch(std::vector<double>& in) {
    InputDataIntoLayers(in.data(), NULL, NULL);
    neural_net->Forward(nullptr);

    auto outputs = new std::vector<double>(kMinibatchSize * size_motors);
    const auto actions_blob = neural_net->blob_by_name(actions_blob_name);
    uint i=0;
    for (uint n = 0; n < kMinibatchSize; ++n)
      for(uint j=0; j < size_motors; j++)
        outputs->at(i++) = actions_blob->data_at(n, j, 0, 0);

    return outputs;
  }

  double weight_l1_norm() {
    double sum = 0.f;

    caffe::Net<double>& net = *neural_net;
    const double* weights;
    for (uint i = 0; i < net.learnable_params().size(); ++i) {
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
      for(int n=0; n < blob->count(); n++)
        sum += fabs(weights[n]);
    }

    return sum;
  }

  double error() {
    auto blob = neural_net->blob_by_name(MLP::loss_blob_name);
    double sum = 0.f;
    const double* errors;
#ifdef CAFFE_CPU_ONLY
    errors = blob->cpu_data();
#else
    errors = caffe::Caffe::mode() == caffe::Caffe::CPU ? blob->cpu_data() : blob->gpu_data();
#endif
    for(int n=0; n < blob->count(); n++)
      sum += fabs(errors[n]);

    return sum;
  }

  boost::shared_ptr<caffe::Net<double>> getNN() {
    return neural_net;
  }

  caffe::Solver<double>* getSolver() {
    return solver;
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
    // PopulateLayer(layer, name, type, bottoms, tops);
    if (include_phase) {
      layer.add_include()->set_phase(*include_phase);
    }
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
  void SliceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const int axis,
                  const std::vector<int>& slice_points) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
    caffe::SliceParameter* slice_param = layer.mutable_slice_param();
    slice_param->set_axis(axis);
    for (auto& p : slice_points) {
      slice_param->add_slice_point(p);
    }
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
  void ReluLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
    caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
    relu_param->set_negative_slope(0.01);
  }
  void TanhLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "TanH", bottoms, tops, include_phase);
  }
  void EltwiseLayer(caffe::NetParameter& net_param,
                    const std::string& name,
                    const std::vector<std::string>& bottoms,
                    const std::vector<std::string>& tops,
                    const boost::optional<caffe::Phase>& include_phase,
                    const caffe::EltwiseParameter::EltwiseOp& op) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Eltwise", bottoms, tops, include_phase);
    caffe::EltwiseParameter* eltwise_param = layer.mutable_eltwise_param();
    eltwise_param->set_operation(op);
  }
  void DummyDataLayer(caffe::NetParameter& net_param,
                      const std::string& name,
                      const std::vector<std::string>& tops,
                      const boost::optional<caffe::Phase>& include_phase,
                      const std::vector<std::vector<double> > shapes,
                      const std::vector<double> values) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "DummyData", {}, tops, include_phase);
    caffe::DummyDataParameter* param = layer.mutable_dummy_data_param();
    for (uint i=0; i<values.size(); ++i) {
      caffe::BlobShape* shape = param->add_shape();
      for (uint j=0; j<shapes[i].size(); ++j) {
        shape->add_dim(shapes[i][j]);
      }
      caffe::FillerParameter* filler = param->add_data_filler();
      filler->set_type("constant");
      filler->set_value(values[i]);
    }
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
    // caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
    // bias_filler->set_type("constant");
    // bias_filler->set_value(1);
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
  void TileLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase,
                 const int axis,
                 const int tiles) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Tile", bottoms, tops, include_phase);
    caffe::TileParameter* param = layer.mutable_tile_param();
    param->set_axis(axis);
    param->set_tiles(tiles);
  }


  std::string Tower(caffe::NetParameter& np,
                    const std::string& layer_prefix,
                    const std::string& input_blob_name,
                    const std::vector<uint>& layer_sizes,
                    uint batch_norm) {
    std::string input_name = input_blob_name;
    for (uint i=1; i<layer_sizes.size()+1; ++i) {
      if(batch_norm == 1) {
        std::string layer_name2 = layer_prefix + "bn" + std::to_string(i);
        BatchNormLayer(np, layer_name2, {input_name}, {layer_name2}, boost::none);
        std::string layer_name3 = layer_prefix + "sc" + std::to_string(i);
        input_name = input_name+"_sc";
        ScaleLayer(np, layer_name3, {layer_name2}, {input_name}, boost::none);
      } else if(batch_norm == 2) {
        std::string input_name2 = input_name+"_bn";
        std::string layer_name2 = layer_prefix + "bn" + std::to_string(i);
        BatchNormLayer(np, layer_name2, {input_name}, {input_name2}, boost::none);
        input_name = input_name2;
      }

      std::string layer_name = layer_prefix + "ip" + std::to_string(i) + "_layer";
      std::string top_name = layer_prefix + "ip" + std::to_string(i);
      IPLayer(np, layer_name, {input_name}, {top_name}, boost::none, layer_sizes[i-1]);
      layer_name = layer_prefix + "ip" + std::to_string(i) + "_relu_layer";
      ReluLayer(np, layer_name, {top_name}, {top_name}, boost::none);
      input_name = top_name;
    }
    return input_name;
  }

  void BatchNormLayer(caffe::NetParameter& net_param,
                      const std::string& name,
                      const std::vector<std::string>& bottoms,
                      const std::vector<std::string>& tops,
                      const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "BatchNorm", bottoms, tops, include_phase);
//     caffe::BatchNormParameter* param = layer.mutable_batch_norm_param();
  }

  void ScaleLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, name, "Scale", bottoms, tops, include_phase);
  }

  void InputDataIntoLayers(double* states_input, double* actions_input, double* target_input) {
    caffe::Net<double>& net = *neural_net;
    if (states_input != NULL) {
      const auto state_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(net.layer_by_name(state_input_layer_name));
//       LOG_DEBUG(state_input_layer->batch_size() << " " << state_input_layer->channels()<< " " << state_input_layer->height()<< " " << state_input_layer->width()<< " " );
      CHECK(state_input_layer);

      state_input_layer->Reset(states_input, states_input, state_input_layer->batch_size());
    }
    if (actions_input != NULL) {
      const auto action_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(
          net.layer_by_name(action_input_layer_name));
      CHECK(action_input_layer);
      action_input_layer->Reset(actions_input, actions_input,
                                action_input_layer->batch_size());
    }
    if (target_input != NULL) {
      const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(
          net.layer_by_name(target_input_layer_name));
      CHECK(target_input_layer);
      target_input_layer->Reset(target_input, target_input,
                                target_input_layer->batch_size());
    }
  }

  void setWeightedSampleVector(double *wsample_input) {
    caffe::Net<double>& net = *neural_net;
    const auto wsample_input_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<double>>(
                                       net.layer_by_name(wsample_input_layer_name));
    CHECK(wsample_input_layer);
    wsample_input_layer->Reset(wsample_input, wsample_input,
                               wsample_input_layer->batch_size());
  }

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

  int GetLayerIndex(const std::string& layer_name) {
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

  void save(const std::string& path) {
    solver->Snapshot();
    caffe::NetParameter net_param;
    neural_net->ToProto(&net_param);
    caffe::WriteProtoToBinaryFile(net_param, path);
  }

  void load(const std::string& path) {
    solver->Restore(path.c_str());
    neural_net->CopyTrainedLayersFrom(path);
  }

 protected:
  caffe::Solver<double>* solver;
  boost::shared_ptr<caffe::Net<double>> neural_net;
  uint size_input_state;
  uint size_sensors;
  uint size_motors;
  uint kMinibatchSize;
  bool add_loss_layer=false;
  bool weighted_sample;
};

#endif  // MLP_HPP


