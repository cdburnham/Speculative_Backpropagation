#include "nn_network.h"
#include "nn_loss.h"
#include <string.h>

static void copy_vec(nn_scalar_t* dst, const nn_scalar_t* src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

nn_status_t nn_net_create(nn_net_t* net) {
    if (!net) return NN_ERR_BADARG;
    memset(net, 0, sizeof(*net));
    return NN_OK;
}

void nn_net_free(nn_net_t* net) {
    if (!net) return;
    memset(net, 0, sizeof(*net));
}

nn_status_t nn_net_add_dense(nn_net_t* net, int in_dim, int out_dim, nn_activation_t act) {
    if (!net || in_dim <= 0 || out_dim <= 0) return NN_ERR_BADARG;
    if (net->num_layers >= nn_g_max_layers) return NN_ERR_UNSUPPORTED;

    if (in_dim > nn_g_max_width || out_dim > nn_g_max_width) return NN_ERR_UNSUPPORTED;

    if (net->num_layers == 0) {
        net->input_dim = in_dim;
    } else {
        int prev_out = net->layers[net->num_layers - 1].out_dim;
        if (in_dim != prev_out) return NN_ERR_SHAPE;
    }

    nn_status_t st = nn_dense_config(&net->layers[net->num_layers], in_dim, out_dim, act);
    if (st != NN_OK) return st;

    net->num_layers++;
    net->output_dim = out_dim;
    return NN_OK;
}

void nn_net_init(nn_net_t* net, nn_init_t init, uint32_t seed) {
    if (!net) return;
    for (int i = 0; i < net->num_layers; i++) {
        nn_dense_init(&net->layers[i], init, seed + (uint32_t)(i * 101u + 7u));
    }
}

nn_status_t nn_net_forward(nn_net_t* net, const nn_scalar_t* x, nn_scalar_t* out) {
    if (!net || !x || !out) return NN_ERR_BADARG;
    if (net->num_layers <= 0) return NN_ERR_BADARG;

    const nn_scalar_t* cur_in = x;

    for (int li = 0; li < net->num_layers; li++) {
        nn_dense_t* L = &net->layers[li];
        copy_vec(net->fwd_inputs[li], cur_in, L->in_dim);
        nn_dense_forward(L, cur_in, net->fwd_outputs[li]);
        cur_in = net->fwd_outputs[li];
    }

    copy_vec(out, net->fwd_outputs[net->num_layers - 1], net->output_dim);
    return NN_OK;
}

nn_status_t nn_net_backward_with_cache(nn_net_t* net,
                                       nn_loss_t loss,
                                       const nn_scalar_t cached_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t cached_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t* target,
                                       nn_scalar_t* out_loss) {
    if (!net || !target) return NN_ERR_BADARG;
    if (net->num_layers <= 0) return NN_ERR_BADARG;

    int od = net->output_dim;
    nn_scalar_t d_buf_a[NN_CAP_MAX_WIDTH];
    nn_scalar_t d_buf_b[NN_CAP_MAX_WIDTH];

    if (out_loss) {
        *out_loss = nn_loss_forward(loss, cached_outputs[net->num_layers - 1], target, od);
    }
    nn_loss_backward(loss, cached_outputs[net->num_layers - 1], target, od, d_buf_a);

    nn_scalar_t* d_cur = d_buf_a;
    nn_scalar_t* d_next = d_buf_b;

    for (int li = net->num_layers - 1; li >= 0; li--) {
        nn_dense_backward(&net->layers[li], cached_inputs[li], cached_outputs[li], d_cur, d_next);

        nn_scalar_t* tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }

    return NN_OK;
}

nn_status_t nn_net_backward_cached(nn_net_t* net,
                                   nn_loss_t loss,
                                   const nn_scalar_t* target,
                                   nn_scalar_t* out_loss) {
    return nn_net_backward_with_cache(net, loss, net->fwd_inputs, net->fwd_outputs, target, out_loss);
}

void nn_net_sgd_step(nn_net_t* net, const nn_sgd_t* opt) {
    if (!net || !opt) return;
    for (int i = 0; i < net->num_layers; i++) {
        nn_dense_sgd_step(&net->layers[i], opt);
    }
}
