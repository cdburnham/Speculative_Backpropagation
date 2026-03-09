#ifndef NN_LAYER_DENSE_H
#define NN_LAYER_DENSE_H

#include "nn_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int in_dim;
    int out_dim;
    nn_activation_t act;

    nn_scalar_t W[NN_CAP_MAX_WIDTH][NN_CAP_MAX_WIDTH];
    nn_scalar_t b[NN_CAP_MAX_WIDTH];

    nn_scalar_t dW[NN_CAP_MAX_WIDTH][NN_CAP_MAX_WIDTH];
    nn_scalar_t db[NN_CAP_MAX_WIDTH];

    nn_scalar_t delta[NN_CAP_MAX_WIDTH];
} nn_dense_t;

nn_status_t nn_dense_config(nn_dense_t* L, int in_dim, int out_dim, nn_activation_t act);
void nn_dense_zero_grads(nn_dense_t* L);
void nn_dense_init(nn_dense_t* L, nn_init_t init, uint32_t seed);
void nn_dense_forward(const nn_dense_t* L, const nn_scalar_t* x, nn_scalar_t* out);
void nn_dense_backward(nn_dense_t* L,
                       const nn_scalar_t* x_prev,
                       const nn_scalar_t* y_post,
                       const nn_scalar_t* d_out,
                       nn_scalar_t* d_in);
void nn_dense_sgd_step(nn_dense_t* L, const nn_sgd_t* opt);

#ifdef __cplusplus
}
#endif

#endif
