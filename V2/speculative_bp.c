#include <stddef.h>
#include "fx16.h"
#include "speculative_bp.h"

#define SPEC_THRESHOLD  FX_FROM_FLOAT(0.02f)

/* ---------------------------------------
   Activation drift gate
--------------------------------------- */
fx16_t activation_drift(layer_t *l) {
    fx16_t sum = 0;
    for(int o=0;o<l->out_size;o++) {
        fx16_t d = fx_abs(fx_sub(l->output[o], l->prev_output[o]));
        sum = fx_add(sum, d);
    }
    return fx_div(sum, fx_from_int(l->out_size));
}

/* ---------------------------------------
   Speculative backward pass
--------------------------------------- */
void network_backward_spec(network_t *n,
                            const fx16_t *input,
                            const fx16_t *target)
{
    int L = n->num_layers - 1;
    layer_t *last = &n->layers[L];

    /* ---- output layer error (prev activations) ---- */
    for(int o=0;o<last->out_size;o++) {
        fx16_t y = last->prev_output[o];
        fx16_t t = target[o];
        last->grad_out[o] = fx_sub(y, t);
    }

    /* ---- backprop through layers ---- */
    for(int i=L;i>=0;i--) {
        layer_t *l = &n->layers[i];
        layer_t *lp = (i>0) ? &n->layers[i-1] : NULL;

        const fx16_t *inp =
            (i==0) ? input : lp->prev_output;

        /* delta + gradients */
        for(int o=0;o<l->out_size;o++) {
            fx16_t dact =
                activation_derivative(l->activation,
                                      l->prev_output[o]);

            fx16_t delta = fx_mul(l->grad_out[o], dact);

            l->grad_b_spec[o] = delta;

            for(int j=0;j<l->in_size;j++) {
                l->grad_w_spec[o][j] =
                    fx_mul(delta, inp[j]);
            }
        }

        /* propagate error backward */
        if(i>0) {
            for(int j=0;j<lp->out_size;j++) {
                fx16_t acc = 0;
                for(int o=0;o<l->out_size;o++) {
                    acc = fx_add(acc,
                        fx_mul(l->weights[o][j],
                               l->grad_out[o]));
                }
                lp->grad_out[j] = acc;
            }
        }
    }
}

/* ---------------------------------------
   Apply speculative gradients
--------------------------------------- */
void network_apply_spec_gradients(network_t *n, fx16_t lr) {
    for(int k=0;k<n->num_layers;k++) {
        layer_t *l = &n->layers[k];

        for(int o=0;o<l->out_size;o++) {
            l->biases[o] =
                fx_sub(l->biases[o],
                       fx_mul(lr, l->grad_b_spec[o]));

            for(int i=0;i<l->in_size;i++) {
                l->weights[o][i] =
                    fx_sub(l->weights[o][i],
                           fx_mul(lr, l->grad_w_spec[o][i]));
            }
        }
    }
}

/* ---------------------------------------
   Speculative forward + commit
--------------------------------------- */
void speculative_forward_commit(network_t *n,
                                const fx16_t *input,
                                const fx16_t *target,
                                fx16_t lr)
{
    /* forward pass */
    network_forward(n, input);

    /* gate on activation drift */
    fx16_t drift =
        activation_drift(&n->layers[n->num_layers-1]);

    if(drift < SPEC_THRESHOLD) {
        network_backward_spec(n, input, target);
        network_apply_spec_gradients(n, lr);
    } else {
        network_backward(n, input, target, lr);
    }

    /* commit snapshot */
    network_snapshot(n);
}