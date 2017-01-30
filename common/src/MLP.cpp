#include "nn/MLP.hpp"

// Layer names
const std::string MLP::state_input_layer_name         = "state_input_layer";
const std::string MLP::action_input_layer_name        = "action_input_layer";
const std::string MLP::target_input_layer_name        = "target_input_layer";
const std::string MLP::wsample_input_layer_name       = "wsample_input_layer";

// Blob names
const std::string MLP::states_blob_name               = "states";
const std::string MLP::actions_blob_name              = "actions";
const std::string MLP::states_actions_blob_name       = "states_actions";
const std::string MLP::targets_blob_name              = "target";
const std::string MLP::wsample_blob_name              = "wsample";
const std::string MLP::q_values_blob_name             = "q_values";
const std::string MLP::loss_blob_name                 = "loss";

const std::string MLP::task_name                      = "task";
