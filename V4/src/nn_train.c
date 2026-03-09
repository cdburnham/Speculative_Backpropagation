#include "nn_train.h"
#include <stdio.h>

static inline const nn_scalar_t* sample_x(const nn_scalar_t* X, int idx, int in_dim) {
    return &X[(size_t)idx * (size_t)in_dim];
}

static inline const nn_scalar_t* sample_y(const nn_scalar_t* Y, int idx, int out_dim) {
    return &Y[(size_t)idx * (size_t)out_dim];
}

static void copy_cache(nn_scalar_t dst[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                       const nn_scalar_t src[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                       const nn_net_t* net,
                       int is_input_cache) {
    for (int li = 0; li < net->num_layers; li++) {
        int n = is_input_cache ? net->layers[li].in_dim : net->layers[li].out_dim;
        for (int i = 0; i < n; i++) {
            dst[li][i] = src[li][i];
        }
    }
}

nn_status_t nn_train_sgd(nn_net_t* net,
                         nn_loss_t loss,
                         const nn_scalar_t* X,
                         const nn_scalar_t* Y,
                         int samples,
                         int epochs,
                         const nn_sgd_t* opt,
                         int print_every) {
    if (!net || !X || !Y || !opt || samples <= 0 || epochs <= 0) return NN_ERR_BADARG;

    nn_scalar_t pred[NN_CAP_MAX_WIDTH];

    for (int e = 1; e <= epochs; e++) {
        nn_scalar_t epoch_loss = (nn_scalar_t)0;

        for (int n = 0; n < samples; n++) {
            const nn_scalar_t* x = sample_x(X, n, net->input_dim);
            const nn_scalar_t* y = sample_y(Y, n, net->output_dim);

            nn_status_t st = nn_net_forward(net, x, pred);
            if (st != NN_OK) return st;

            nn_scalar_t loss_sample = (nn_scalar_t)0;
            st = nn_net_backward_cached(net, loss, y, &loss_sample);
            if (st != NN_OK) return st;

            nn_net_sgd_step(net, opt);
            epoch_loss += loss_sample;
        }

        epoch_loss /= (nn_scalar_t)samples;
        if (print_every > 0 && (e % print_every) == 0) {
            printf("Epoch %d  Loss=%.6f\n", e, (double)epoch_loss);
        }
    }

    return NN_OK;
}

nn_status_t nn_train_sgd_spec1(nn_net_t* net,
                               nn_loss_t loss,
                               const nn_scalar_t* X,
                               const nn_scalar_t* Y,
                               int samples,
                               int epochs,
                               const nn_sgd_t* opt,
                               int print_every) {
    if (!net || !X || !Y || !opt || samples <= 0 || epochs <= 0) return NN_ERR_BADARG;

    nn_scalar_t pred[NN_CAP_MAX_WIDTH];
    nn_scalar_t prev_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t prev_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t curr_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t curr_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t y_prev[NN_CAP_MAX_WIDTH] = {0};

    for (int e = 1; e <= epochs; e++) {
        nn_scalar_t epoch_loss = (nn_scalar_t)0;

        const nn_scalar_t* x0 = sample_x(X, 0, net->input_dim);
        const nn_scalar_t* t0 = sample_y(Y, 0, net->output_dim);
        nn_status_t st = nn_net_forward(net, x0, pred);
        if (st != NN_OK) return st;

        copy_cache(prev_inputs, net->fwd_inputs, net, 1);
        copy_cache(prev_outputs, net->fwd_outputs, net, 0);
        for (int i = 0; i < net->output_dim; i++) {
            y_prev[i] = t0[i];
        }

        for (int t = 1; t < samples; t++) {
            nn_scalar_t loss_prev = (nn_scalar_t)0;
            st = nn_net_backward_with_cache(net, loss, prev_inputs, prev_outputs, y_prev, &loss_prev);
            if (st != NN_OK) return st;

            nn_net_sgd_step(net, opt);
            epoch_loss += loss_prev;

            const nn_scalar_t* xt = sample_x(X, t, net->input_dim);
            const nn_scalar_t* yt = sample_y(Y, t, net->output_dim);
            st = nn_net_forward(net, xt, pred);
            if (st != NN_OK) return st;

            copy_cache(curr_inputs, net->fwd_inputs, net, 1);
            copy_cache(curr_outputs, net->fwd_outputs, net, 0);

            copy_cache(prev_inputs, curr_inputs, net, 1);
            copy_cache(prev_outputs, curr_outputs, net, 0);
            for (int i = 0; i < net->output_dim; i++) {
                y_prev[i] = yt[i];
            }
        }

        nn_scalar_t loss_last = (nn_scalar_t)0;
        st = nn_net_backward_with_cache(net, loss, prev_inputs, prev_outputs, y_prev, &loss_last);
        if (st != NN_OK) return st;

        nn_net_sgd_step(net, opt);
        epoch_loss += loss_last;

        epoch_loss /= (nn_scalar_t)samples;
        if (print_every > 0 && (e % print_every) == 0) {
            printf("Epoch %d  Loss=%.6f\n", e, (double)epoch_loss);
        }
    }

    return NN_OK;
}
