#include "nn_layer_dense.h"
#include "nn_activation.h"
#include "nn_math.h"
#include "nn_rng.h"
#include <string.h>

nn_status_t nn_dense_config(nn_dense_t* L, int in_dim, int out_dim, nn_activation_t act) {
    if (!L || in_dim <= 0 || out_dim <= 0) return NN_ERR_BADARG;
    if (in_dim > nn_g_max_width || out_dim > nn_g_max_width) return NN_ERR_UNSUPPORTED;

    memset(L, 0, sizeof(*L));
    L->in_dim = in_dim;
    L->out_dim = out_dim;
    L->act = act;
    return NN_OK;
}

void nn_dense_zero_grads(nn_dense_t* L) {
    if (!L) return;
    for (int o = 0; o < L->out_dim; o++) {
        L->db[o] = (nn_scalar_t)0;
        for (int i = 0; i < L->in_dim; i++) {
            L->dW[o][i] = (nn_scalar_t)0;
        }
    }
}

void nn_dense_init(nn_dense_t* L, nn_init_t init, uint32_t seed) {
    if (!L) return;

    int fan_in = L->in_dim;
    int fan_out = L->out_dim;
    nn_scalar_t scale = (nn_scalar_t)1;
    uint32_t s = seed ? seed : 1u;

    switch (init) {
        case NN_INIT_UNIFORM_SYM:
            scale = (nn_scalar_t)1;
            break;
        case NN_INIT_XAVIER:
            scale = nn_sqrt((nn_scalar_t)6 / (nn_scalar_t)(fan_in + fan_out));
            break;
        case NN_INIT_HE:
            scale = nn_sqrt((nn_scalar_t)6 / (nn_scalar_t)fan_in);
            break;
        default:
            scale = (nn_scalar_t)1;
            break;
    }

    for (int o = 0; o < L->out_dim; o++) {
        for (int i = 0; i < L->in_dim; i++) {
            L->W[o][i] = nn_rng_uniform_sym(&s, scale);
        }
        L->b[o] = nn_rng_uniform_sym(&s, scale);
    }
}

void nn_dense_forward(const nn_dense_t* L, const nn_scalar_t* x, nn_scalar_t* out) {
    int in = L->in_dim;
    int out_dim = L->out_dim;

    for (int o = 0; o < out_dim; o++) {
        nn_scalar_t acc = L->b[o];
        for (int i = 0; i < in; i++) {
            acc += L->W[o][i] * x[i];
        }
        out[o] = acc;
    }

    nn_act_forward(L->act, out, out_dim);
}

void nn_dense_backward(nn_dense_t* L,
                       const nn_scalar_t* x_prev,
                       const nn_scalar_t* y_post,
                       const nn_scalar_t* d_out,
                       nn_scalar_t* d_in) {
    int in = L->in_dim;
    int out = L->out_dim;

    for (int o = 0; o < out; o++) {
        L->delta[o] = d_out[o];
    }
    nn_act_backward_mul(L->act, y_post, L->delta, out);

    for (int o = 0; o < out; o++) {
        nn_scalar_t d = L->delta[o];
        L->db[o] = d;
        for (int i = 0; i < in; i++) {
            L->dW[o][i] = d * x_prev[i];
        }
    }

    for (int i = 0; i < in; i++) {
        nn_scalar_t acc = (nn_scalar_t)0;
        for (int o = 0; o < out; o++) {
            acc += L->W[o][i] * L->delta[o];
        }
        d_in[i] = acc;
    }
}

void nn_dense_sgd_step(nn_dense_t* L, const nn_sgd_t* opt) {
    if (!L || !opt) return;

    nn_scalar_t lr = opt->lr;
    for (int o = 0; o < L->out_dim; o++) {
        L->b[o] -= lr * L->db[o];
        for (int i = 0; i < L->in_dim; i++) {
            L->W[o][i] -= lr * L->dW[o][i];
        }
    }
}
