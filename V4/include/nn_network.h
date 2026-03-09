#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn_layer_dense.h"
#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_layers;
    int input_dim;
    int output_dim;

    nn_dense_t layers[NN_CAP_MAX_LAYERS];

    nn_scalar_t fwd_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH];
    nn_scalar_t fwd_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH];
} nn_net_t;

nn_status_t nn_net_create(nn_net_t* net);
void nn_net_free(nn_net_t* net);
nn_status_t nn_net_add_dense(nn_net_t* net, int in_dim, int out_dim, nn_activation_t act);
void nn_net_init(nn_net_t* net, nn_init_t init, uint32_t seed);

nn_status_t nn_net_forward(nn_net_t* net, const nn_scalar_t* x, nn_scalar_t* out);

nn_status_t nn_net_backward_cached(nn_net_t* net,
                                   nn_loss_t loss,
                                   const nn_scalar_t* target,
                                   nn_scalar_t* out_loss);

nn_status_t nn_net_backward_with_cache(nn_net_t* net,
                                       nn_loss_t loss,
                                       const nn_scalar_t cached_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t cached_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t* target,
                                       nn_scalar_t* out_loss);

void nn_net_sgd_step(nn_net_t* net, const nn_sgd_t* opt);

#ifdef __cplusplus
}
#endif

#endif
