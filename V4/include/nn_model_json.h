#ifndef NN_MODEL_JSON_H // include guard
#define NN_MODEL_JSON_H

#include "nn_network.h" // include nn_net_t and related functions

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// Model configuration structure and functions
typedef struct {
    int out_dim;
    nn_activation_t act;
} nn_layer_cfg_t;

// This structure holds all the configuration parameters needed to build and train a network, and can be loaded from a JSON file.
typedef struct {
    int max_layers;
    int max_width;

    int input_dim;
    int num_layers;
    nn_layer_cfg_t layers[NN_CAP_MAX_LAYERS];

    nn_init_t init;
    uint32_t seed;

    nn_loss_t loss;
    nn_sgd_t opt;
    int epochs;
    int print_every;
    int use_spec1;
} nn_model_config_t;

// PROTOTYPES:

// Fills a config structure with default values.
// You can modify these defaults by loading from a JSON file or changing the fields directly.
void nn_model_config_default(nn_model_config_t* cfg);

// Loads a config from a JSON file. 
// Returns NN_ERR_BADARG if the file is invalid or constraints are violated.
nn_status_t nn_model_config_load_json(const char* path, nn_model_config_t* cfg);

// Builds a network from a config structure.
// Returns NN_ERR_BADARG if the config is invalid or constraints are violated.
nn_status_t nn_model_build_from_config(const nn_model_config_t* cfg, nn_net_t* net);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard