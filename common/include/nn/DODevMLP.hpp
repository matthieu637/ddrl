#ifndef DODevMLP_HPP
#define DODevMLP_HPP

#include "MLP.hpp"
#include "bib/IniParser.hpp"

#define LOWER_REWARD -1e8

class DODevMLP : public MLP {
 public:

  DODevMLP(unsigned int input, unsigned int sensors, const std::vector<uint>& hiddens, double alpha,
           uint _kMinibatchSize, double decay_v, uint hidden_layer_type, uint batch_norm, bool _weighted_sample=false,
           uint mom_=0) :
    MLP(input, sensors, hiddens, alpha, _kMinibatchSize, decay_v, hidden_layer_type, batch_norm, _weighted_sample, mom_) {

  }

  //   policy
  DODevMLP(unsigned int sensors, const std::vector<uint>& hiddens, unsigned int motors, double alpha,
           uint _kMinibatchSize,
           uint hidden_layer_type, uint last_layer_type, uint batch_norm, bool loss_layer=false, uint mom_=0) :
    MLP(sensors, hiddens, motors, alpha, _kMinibatchSize, hidden_layer_type, last_layer_type, batch_norm, loss_layer,
        mom_) {

  }

  DODevMLP(const MLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) :
    DODevMLP(static_cast<const DODevMLP&>(m), copy_solver, _phase) {
  }

  DODevMLP(const DODevMLP& m, bool copy_solver, ::caffe::Phase _phase = ::caffe::Phase::TRAIN) :
    MLP(m, copy_solver, _phase), disable_st_control(m.disable_st_control),
    disable_ac_control(m.disable_ac_control), heuristic(m.heuristic),
    reset_learning_algo(m.reset_learning_algo), intrasec_motivation(m.intrasec_motivation),
    im_window(m.im_window), im_smooth(m.im_smooth), im_index(m.im_index), ewc(m.ewc),
    last_episode_changed(m.last_episode_changed), informed_sensorimotor_space(m.informed_sensorimotor_space){
    st_control = new std::vector<uint>(*m.st_control);
    ac_control = new std::vector<uint>(*m.ac_control);
    if(heuristic == 1)
      heuristic_devpoints = new std::vector<uint>(*m.heuristic_devpoints);
    else if(heuristic == 2)
      heuristic_linearcoef = new std::vector<double>(*m.heuristic_linearcoef);
    ac_informed_state = new std::vector<uint>(*m.ac_informed_state);
    if(ewc >= 0.f)
      LOG_WARNING("be careful not sure it's working");
    if(informed_sensorimotor_space)
      pd_controller_factor = new std::vector<double>(*m.pd_controller_factor);
  }

  virtual ~DODevMLP() {
    if(st_control != nullptr)
      delete st_control;
    if(ac_control != nullptr)
      delete ac_control;
    if(heuristic_devpoints != nullptr)
      delete heuristic_devpoints;
    if(heuristic_linearcoef != nullptr)
      delete heuristic_linearcoef;
    if(best_param != nullptr)
      delete best_param;
    if(best_param_previous_task != nullptr)
      delete best_param_previous_task;
    if(fisher != nullptr)
      delete fisher;
    if(previous_fisher != nullptr)
      delete previous_fisher;
    if(ac_informed_state != nullptr)
      delete ac_informed_state;
    if(pd_controller_factor != nullptr)
      delete pd_controller_factor;
  }

  virtual void exploit(boost::property_tree::ptree* pt, MLP* actor) override {
    bool st_scale = pt->get<bool>("devnn.st_scale");
    bool ac_scale = pt->get<bool>("devnn.ac_scale");
    uint st_probabilistic = pt->get<uint>("devnn.st_probabilistic");
    uint ac_probabilistic = pt->get<uint>("devnn.ac_probabilistic");
    bool compute_diff_backward = false;
    st_control = bib::to_array<uint>(pt->get<std::string>("devnn.st_control"));
    ac_control = bib::to_array<uint>(pt->get<std::string>("devnn.ac_control"));
    heuristic = 0;
    intrasec_motivation = false;
    ewc = -1.f;
    ewc_decay = -1.f;
    try {
      heuristic = pt->get<uint>("devnn.heuristic");
      if((heuristic == 1 && st_probabilistic != 1) || (heuristic == 1 && ac_probabilistic != 1)) {
        LOG_INFO("heuristic is 1 so probabilistic should be 1");
        exit(1);
      }
      if((heuristic == 2 && st_probabilistic != 0) || (heuristic == 2 && ac_probabilistic != 0)) {
        LOG_INFO("heuristic is 2 so probabilistic should be 0");
        exit(1);
      }
      LOG_INFO("catch heuristic " <<heuristic);
    } catch(boost::exception const& ) {
    }

    try {
      intrasec_motivation = pt->get<bool>("devnn.intrasec_motivation");
      if(intrasec_motivation && heuristic != 1) {
        LOG_INFO("intrasec motiv works only with determinist activation and heuristic = 1");
        exit(1);
      }
    } catch(boost::exception const& ) {
    }

    if(intrasec_motivation) {
      im_smooth = pt->get<int>("devnn.im_smooth");
      im_window = pt->get<int>("devnn.im_window");
    }

    if(heuristic == 1) {
      try {
        heuristic_devpoints = bib::to_array<uint>(pt->get<std::string>("devnn.heuristic_devpoints"));
        ASSERT((st_control->size() + ac_control->size()) == heuristic_devpoints->size(), "st " << st_control->size()
               << " " << ac_control->size() << " " << heuristic_devpoints->size() );
      } catch(boost::exception const& ) {
      }
    } else if(heuristic == 2) {
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

    try {
      compute_diff_backward = pt->get<bool>("devnn.compute_diff_backward");
    } catch(boost::exception const& ) {
    }

    try {
      reset_learning_algo = pt->get<bool>("devnn.reset_learning_algo");
    } catch(boost::exception const& ) {
    }
    informed_sensorimotor_space = false;
    try {
      informed_sensorimotor_space = pt->get<bool>("devnn.informed_sensorimotor_space");
    } catch(boost::exception const& ) {
    }
    if(informed_sensorimotor_space && !isCritic()){
      ac_informed_state = bib::to_array<uint>(pt->get<std::string>("devnn.ac_informed_state"));
      pd_controller_factor = bib::to_array<double>(pt->get<std::string>("devnn.pd_controller_factor"));
      
      if(ac_informed_state->size() != (uint) size_motors * 2){
        LOG_ERROR("wrong ac_informed_state " << ac_informed_state->size() );
        exit(1);
      }
    }
    
    if(informed_sensorimotor_space && ac_probabilistic ==0){
      LOG_ERROR("informed_sensorimotor_space is not implemented with proba action");
      exit(1);
    }
    
    if(informed_sensorimotor_space && heuristic == 2){
      LOG_ERROR("informed_sensorimotor_space is not implemented with proba action");
      exit(1);
    }

    try {
      ewc = pt->get<double>("devnn.ewc");
    } catch(boost::exception const& ) {
    }
    
    try {
      if(ewc >= 0.f)
        ewc_decay = pt->get<double>("devnn.ewc_decay");
    } catch(boost::exception const& ) {
    }

    if(ewc >= 0.f){
      ewc_fisher_beta           = pt->get<double>("devnn.ewc_fisher_beta");
      if(ewc_fisher_beta < 0){
        LOG_ERROR("ewc_fisher_beta should be between [0;1]");
        exit(1);
      }
      ewc_best_param_method     = pt->get<uint>("devnn.ewc_best_param_method");
      ewc_for_critic            = pt->get<uint>("devnn.ewc_for_critic");
    }
    
    if((ac_scale && ac_probabilistic != 0) || (st_scale && st_probabilistic != 0)) {
      LOG_ERROR("if not probabilistic then why scaling? of how much?");
      exit(1);
    }

    if(heuristic == 2 && reset_learning_algo) {
      LOG_ERROR("when sould I reset ? I'm progressive");
      exit(1);
    }

    if(heuristic == 0 && reset_learning_algo) {
      LOG_ERROR("when sould I reset, I'm learning alone all my parameters");
      exit(1);
    }

    if(heuristic != 0 && compute_diff_backward) {
      LOG_ERROR("why do you compute diff if you don't use local solver?");
      exit(1);
    }

    if(!reset_learning_algo && ewc >= 0.f) {
      LOG_WARNING("be careful not to make ewc_setup() dependent of inform()");
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

    uint nb_action_layer = 0;
    for(caffe::LayerParameter& lp : *net_param_init.mutable_layer())
      for(const std::string& top : lp.top())
        if(top == MLP::actions_blob_name)
          nb_action_layer++;

    if(nb_action_layer == 0)
      disable_ac_control = true;

//     net_param_init.PrintDebugString();
//     LOG_DEBUG("########################################################################################");
    uint nb_state_layer = 0;
    nb_action_layer = 0;
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
        } else if(top == MLP::states_blob_name && state_capted_bottom && !disable_st_control) {
          *lp.mutable_top(b) = "devnn.states";
        }
        b++;
      }

      if(nb_state_layer > 0 && !state_dev_inserted) {
        caffe::DevelopmentalParameter* dp = addDevState(net_param_new, heuristic);
        dp->set_scale(st_scale);
        dp->set_probabilist(st_probabilistic);
        for (uint c : *st_control)
          dp->add_control(c);
        state_dev_inserted = true;
      }

      if(lp.name() == MLP::loss_blob_name && !action_dev_inserted && !disable_ac_control) {
        caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
        dp->set_scale(ac_scale);
        dp->set_probabilist(ac_probabilistic);
        dp->set_diff_compute(compute_diff_backward);
        for (uint c : *ac_control)
          dp->add_control(c);
        action_dev_inserted = true;
      }

      caffe::LayerParameter* lpc = net_param_new.add_layer();
      lpc->CopyFrom(lp);

      if(nb_action_layer > 0 && !action_dev_inserted && input_action) {
        caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
        dp->set_scale(ac_scale);
        dp->set_probabilist(ac_probabilistic);
        dp->set_diff_compute(compute_diff_backward);
        for (uint c : *ac_control)
          dp->add_control(c);
        action_dev_inserted = true;
      }
    }

    if(nb_action_layer > 0 && !action_dev_inserted && !input_action) {
      caffe::DevelopmentalParameter* dp = addDevAction(net_param_new, input_action, heuristic);
      dp->set_scale(ac_scale);
      dp->set_probabilist(ac_probabilistic);
      dp->set_diff_compute(compute_diff_backward);
      for (uint c : *ac_control)
        dp->add_control(c);
      action_dev_inserted = true;
    }

#ifndef NDEBUG
    if(!disable_st_control)
      ASSERT(nb_state_layer >= 1, "check nb of state layer " << nb_state_layer);
    //not true for vnn
    if(!disable_ac_control)
      ASSERT(nb_action_layer >= 1, "check nb of action layer " << nb_action_layer);
#endif
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

    if(actor != nullptr) {
      ASSERT(net_param_new.name() == "Critic", "net is not a critic " << net_param_new.name());
      ASSERT(actor->getNN()->name() == "Actor", "net is not an actor " << actor->getNN()->name());
      if(!disable_ac_control) {
        neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareData(
          *actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
        neural_net->layer_by_name("devnn_actions")->blobs()[0]->ShareDiff(
          *actor->getNN()->layer_by_name("devnn_actions")->blobs()[0]);
      }

      if(!disable_st_control) {
        neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareData(
          *actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
        neural_net->layer_by_name("devnn_states")->blobs()[0]->ShareDiff(
          *actor->getNN()->layer_by_name("devnn_states")->blobs()[0]);
      }
    }

    if(heuristic != 0) {
      if(!disable_ac_control) {
        auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
        auto count = blob_ac->count();
        auto data = blob_ac->mutable_cpu_data();
        for(int i=0; i<count; i++)
          data[i] = 0.0f;
      }

      if(!disable_st_control) {
        auto blob_ac = neural_net->layer_by_name("devnn_states")->blobs()[0];
        auto count = blob_ac->count();
        auto data = blob_ac->mutable_cpu_data();
        for(int i=0; i<count; i++)
          data[i] = 0.0f;
      }
    }

    if(intrasec_motivation &&
        std::find(heuristic_devpoints->begin(), heuristic_devpoints->end(), 0) != heuristic_devpoints->end()) {
      update_DL_IM();
    }
  }

 private:
  caffe::DevelopmentalParameter* addDevState(caffe::NetParameter& net_param, uint heuristic) {
    caffe::LayerParameter& layer = *net_param.add_layer();
    PopulateLayer(layer, "devnn_states", "Developmental", {MLP::states_blob_name}, {"devnn.states"}, boost::none);
    if(heuristic != 0) {
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
    if(heuristic != 0) {
      caffe::ParamSpec* fixed_param_spec = layer.add_param();
      fixed_param_spec->set_lr_mult(0.f);
      fixed_param_spec->set_decay_mult(0.f);
    }
    return layer.mutable_developmental_param();
  }

  bool going_to_develop(uint episode) {
    if(heuristic == 1) {
      for(uint heuristic_devpoints_index=0; heuristic_devpoints_index < heuristic_devpoints->size() ;
          heuristic_devpoints_index++)
        if(episode == heuristic_devpoints->at(heuristic_devpoints_index)) {
          if(heuristic_devpoints_index < st_control->size()) {
            return true;
          } else if(heuristic_devpoints_index < st_control->size() + ac_control->size() && !disable_ac_control) {
            return true;
          }
        }
    } else if(heuristic == 2)
      return true;

    return false;
  }

  bool develop(uint episode) {
    bool catch_change = false;
    bool will_change = going_to_develop(episode);
    if(will_change)
      ewc_setup(episode);

    if(heuristic == 1) {
      for(uint heuristic_devpoints_index=0; heuristic_devpoints_index < heuristic_devpoints->size() ;
          heuristic_devpoints_index++)
        if(episode == heuristic_devpoints->at(heuristic_devpoints_index)) {
          if(heuristic_devpoints_index < st_control->size()) {
            auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
            auto data = blob_st->mutable_cpu_data();
            data[heuristic_devpoints_index] = 1.f;
            LOG_INFO("dev point st " << st_control->at(heuristic_devpoints_index) );
            catch_change = true;
          } else if(heuristic_devpoints_index < st_control->size() + ac_control->size() && !disable_ac_control) {
            auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
            auto data = blob_ac->mutable_cpu_data();
            data[heuristic_devpoints_index - st_control->size()] = 1.f;
            LOG_INFO("dev point ac " << ac_control->at(heuristic_devpoints_index - st_control->size()) );
            catch_change = true;
          }
        }
    } else if(heuristic == 2) {
      catch_change = true;
      if(!disable_st_control) {
        auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
        auto data_st = blob_st->mutable_cpu_data();
        for(uint i=0; i < st_control->size() ; i++) {
          if( heuristic_linearcoef->at(i) >= 1.)
            data_st[i] = 1.;
          else
            data_st[i] = ((double)episode) * heuristic_linearcoef->at(i);
        }
      }

      if(!disable_ac_control) {
        auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
        auto data = blob_ac->mutable_cpu_data();
        for(uint i=0; i < ac_control->size() ; i++) {
          if( heuristic_linearcoef->at(i + st_control->size()) >= 1.)
            data[i] = 1.;
          else
            data[i] = ((double)episode) * heuristic_linearcoef->at(i + st_control->size());
        }
      }
    }

    return catch_change;
  }

  bool developIM(int episode, double score) {
    if(episode == 0)
      return false;

    bool changed = false;

    episode = episode - last_episode_changed - 1;//because called during starting episode
    //update stats
    all_scores.push_back(score);

    double new_e = 0.f;
    double new_ew = 0.f;
    for(int i=0; i<im_smooth; i++) {
      if(episode - i < 0)
        new_e += LOWER_REWARD;
      else
        new_e += all_scores[episode-i];
      if(episode - i - im_window < 0)
        new_ew += LOWER_REWARD;
      else
        new_ew += all_scores[episode-i-im_window];
    }
    new_e *= 1./(double) im_smooth;
    new_ew *= 1./(double) im_smooth;

    //decide if changed
    changed = new_e - new_ew <= 0.f &&
              episode >= (im_smooth + im_window) &&
              still_something_to_develop();

    //change corresponding weights
    if(changed) {
      ewc_setup(last_episode_changed + all_scores.size());
      update_DL_IM();

      all_scores.clear();
    }

    return changed;
  }

  bool still_something_to_develop() {
    uint heuristic_devpoints_index=0;
    for(auto k : *heuristic_devpoints) {
      if(k <= im_index + 1) {
        if(heuristic_devpoints_index < st_control->size()) {
          auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
          auto data = blob_st->mutable_cpu_data();
          if(data[heuristic_devpoints_index] == 0.f)
            return true;
        } else if(heuristic_devpoints_index < st_control->size() + ac_control->size() && !disable_ac_control) {
          auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
          auto data = blob_ac->mutable_cpu_data();
          if(data[heuristic_devpoints_index - st_control->size()] == 0.f)
            return true;
        }
      }
      heuristic_devpoints_index++;
    }

    return false;
  }

  void update_DL_IM() {
    uint heuristic_devpoints_index=0;
    for(auto k : *heuristic_devpoints) {
      if(k <= im_index) {
        if(heuristic_devpoints_index < st_control->size()) {
          auto blob_st = neural_net->layer_by_name("devnn_states")->blobs()[0];
          auto data = blob_st->mutable_cpu_data();
          data[heuristic_devpoints_index] = 1.f;
          LOG_INFO("dev point st " << st_control->at(heuristic_devpoints_index) );
        } else if(heuristic_devpoints_index < st_control->size() + ac_control->size() && !disable_ac_control) {
          auto blob_ac = neural_net->layer_by_name("devnn_actions")->blobs()[0];
          auto data = blob_ac->mutable_cpu_data();
          data[heuristic_devpoints_index - st_control->size()] = 1.f;
          LOG_INFO("dev point ac " << ac_control->at(heuristic_devpoints_index - st_control->size()) );
        }
      }
      heuristic_devpoints_index++;
    }
    im_index++;
  }

 public:
//   Used by others
  std::tuple<bool, bool> inform(uint episode_, double score) {
    bool changed = false;
    if(intrasec_motivation)
      changed = developIM(episode_, score);
    else
      changed = develop(episode_);
    
    return std::make_tuple(reset_learning_algo, changed);
  }
  
  void soft_update(const MLP& from, double tau) override {
    auto net_from = from.neural_net;
    auto net_to = neural_net;
    
    const auto& from_params = net_from->params();
    const auto& to_params = net_to->params();
    CHECK_EQ(from_params.size(), to_params.size());
    CHECK_EQ(neural_net->params_lr().size(), to_params.size());
    
    for (uint i = 0; i < from_params.size(); ++i) {
      if(neural_net->params_lr()[i] != 0.0f) {
        auto& from_blob = from_params[i];
        auto& to_blob = to_params[i];
        caffe::caffe_cpu_axpby(from_blob->count(), tau, from_blob->cpu_data(),
                               (1.f-tau), to_blob->mutable_cpu_data());
      }
    }
  }
  
  virtual void actor_backward() override {
    int layer = get_layer_index("devnn_actions");
    
    ASSERT(layer != -1, "failed to found layer for backward");
    
    neural_net->BackwardFrom(layer);
  }
  
//   EWC methods
  void ewc_setup(uint episode) {
    if(!ewc_enabled() || best_param ==nullptr)
      return ;
    
    last_episode_changed = episode;
    ewc_decay_multiplier = 1.f;
    best_score = std::numeric_limits<double>::lowest();
    
    if(best_param_previous_task != nullptr)
      delete best_param_previous_task;
    best_param_previous_task = new std::vector<double>(*best_param);
    this->copyWeightsFrom(best_param_previous_task->data(), true);
//     std::string s = " best " + neural_net->name() + " ";
//     bib::Logger::PRINT_ELEMENTS(*best_param_previous_task, s.c_str());
    
    for(uint i=0; i<fisher->size(); i++)
      fisher->at(i) = fisher->at(i) / fisher_nbr;
    
    double fmax_ = *std::max_element(fisher->begin(), fisher->end());
    for(uint i=0; i<fisher->size(); i++)
      fisher->at(i) = fisher->at(i) / fmax_;
    
    if(previous_fisher != nullptr)
      delete previous_fisher;
    previous_fisher = fisher;
    fisher = nullptr;

//     bib::Logger::PRINT_ELEMENTS(*previous_fisher, " fisher ");
  }
  
  void updateFisher(double number_sample) override {
    if(!ewc_enabled())
      return;
    
    if(fisher == nullptr) {
      fisher = new std::vector<double>(number_of_parameters(true), 0.f);
      fisher_nbr = 0.f;
    }
    
    uint k=0;
    ASSERT(neural_net->params_lr().size() == neural_net->params().size(), "size pb");
    for (uint i = 0; i < neural_net->params().size(); ++i) {
      if(neural_net->params_lr()[i] != 0.0f) {
        auto blob = neural_net->params()[i];
        auto derriv = blob->cpu_diff();
        for(int y=0; y <blob->count(); y++) {
          fisher->at(k) = fisher->at(k) * ewc_fisher_beta + (derriv[y] * derriv[y]) * number_sample;
          k++;
        }
      }
    }
    fisher_nbr = fisher_nbr * ewc_fisher_beta + number_sample;
    ASSERT(k == fisher->size(), "size pb");
  }

  double ewc_cost() override {
    if(!ewc_enabled() || last_episode_changed == 0)
      return 0.f;
    double cost = 0.f;
    uint k=0;
    const auto& from_params = neural_net->params();
    ASSERT(previous_fisher->size() == best_param_previous_task->size(), "size pb");
    for (uint i = 0; i < from_params.size(); ++i) {
      if(neural_net->params_lr()[i] != 0.0f) {
        auto& from_blob = from_params[i];
        auto weight = from_blob->cpu_data();
        for(int j=0; j<from_blob->count(); j++) {
          double x = (best_param_previous_task->at(k) - weight[j]);
          cost += x*x*previous_fisher->at(k);
          k++;
        }
      }
    }

    double r = cost * ewc / ((double) k);
    if(ewc_decay < 0.f)
      return r;
    r *= ewc_decay_multiplier;
    return r;
  }

  void regularize() override {
    if(!ewc_enabled() || last_episode_changed == 0)
      return;

    uint k=0;
    double ewc_decay_factor_ = 1.f;
    if(ewc_decay >= 0.f){
      ewc_decay_factor_ = ewc_decay_multiplier;
      ewc_decay_multiplier *= ewc_decay;
    }
      
    const auto& from_params = neural_net->params();
    for (uint i = 0; i < from_params.size(); ++i) {
      if(neural_net->params_lr()[i] != 0.0f) {
        auto& from_blob = from_params[i];
        auto weight = from_blob->cpu_data();
        auto weight_diff = from_blob->mutable_cpu_diff();
        for(int j=0; j<from_blob->count(); j++) {
          weight_diff[j] += ewc * ewc_decay_factor_ * previous_fisher->at(k) * (weight[j] - best_param_previous_task->at(k)) ;
          k++;
        }
      }
    }
  }

  bool ewc_enabled() override {
    bool ewc_var = ewc >= 0.f;
    if(!isCritic()) //actor
      return ewc_var;
    
    return ewc_var && ewc_for_critic >= 1;
  }

  void update_best_param_previous_task(double score) override {
    if(ewc_enabled()) {
      bool going_to_update = false;
      switch (ewc_best_param_method){
        case 3 ://keep a recent "good" performance
        case 2 ://learning from best sample in learning (might never be as good as sample)
        case 0 ://best param in learning
          going_to_update = score > best_score;
          break;
        case 1 ://last
          going_to_update = true;
          break;
      }
      if(going_to_update) {
        best_score = score;

        if(best_param != nullptr)
          delete best_param;

        best_param = new std::vector<double>(number_of_parameters(true));
        this->copyWeightsTo(best_param->data(), true);
      }

      ewc_decay_multiplier *= ewc_decay;
    }
  }
  
  bool ewc_force_constraint(){
    return ewc_for_critic == 2;
  }
//   Informed sensorimotor space
  void neutral_action(const std::vector<double>& state, std::vector<double>* action) override {
    if(informed_sensorimotor_space && ac_control->size() > 0 && !isCritic()){
      auto blob_dev_weights = neural_net->learnable_params()[neural_net->learnable_params().size()-1];
      auto weights = blob_dev_weights->cpu_data();
      ASSERT(blob_dev_weights->count() == ac_control->size(), "pb size");
      for(int i=0;i<blob_dev_weights->count();i++){
        if(weights[i] < 0.5){ //dimension should not be controled
          uint st_index = ac_informed_state->at(ac_control->at(i)*2);
          action->at(ac_control->at(i)) = (2.0f/M_PI) * 
              atan(pd_controller_factor->at(0)*state[st_index] + pd_controller_factor->at(1) * state[st_index+1]) * pd_controller_factor->at(2);
        }
      }
    }
  }

 private:
//    structure of activation
  bool disable_st_control = false;
  bool disable_ac_control = false; //for vnn
  uint heuristic = 0;
  std::vector<uint>* heuristic_devpoints = nullptr;
  std::vector<double>* heuristic_linearcoef = nullptr;
  std::vector<uint>* st_control = nullptr;
  std::vector<uint>* ac_control = nullptr;
  bool reset_learning_algo = false;
//   intrasec motivation
  bool intrasec_motivation;
  int im_window, im_smooth;
  uint im_index = 0;
//   Elastic weight consolidation
  double ewc = -1.f;
  double ewc_decay = -1.f;
  double ewc_decay_multiplier = 1.f;
  double ewc_fisher_beta;
  uint ewc_best_param_method;//0 for best, 1 for last
  uint ewc_for_critic; // 0 no EWC for critic - 1 EWC for critic - 2 EWC for critic forced by actor
//   Informed sensorimotor space
  bool informed_sensorimotor_space;
  std::vector<uint>* ac_informed_state = nullptr;
  std::vector<double>* pd_controller_factor = nullptr;

  std::vector<double> all_scores;
  std::vector<double>* best_param = nullptr;
  std::vector<double>* best_param_previous_task = nullptr;
  std::vector<double>* fisher = nullptr;
  std::vector<double>* previous_fisher = nullptr;
  double fisher_nbr;
  uint last_episode_changed = 0;
  double best_score = std::numeric_limits<double>::lowest();
};

#endif




