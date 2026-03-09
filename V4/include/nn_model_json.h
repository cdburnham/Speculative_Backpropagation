#ifndef NN_MODEL_JSON_H
#define NN_MODEL_JSON_H

#include "nn_network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int out_dim;
    nn_activation_t act;
} nn_layer_cfg_t;

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

void nn_model_config_default(nn_model_config_t* cfg);
nn_status_t nn_model_config_load_json(const char* path, nn_model_config_t* cfg);
nn_status_t nn_model_build_from_config(const nn_model_config_t* cfg, nn_net_t* net);

#ifdef __cplusplus
}
#endif

#endif
