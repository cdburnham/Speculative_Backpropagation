#include "fx16.h"
#include "layer.h"
#include <stdlib.h>

void layer_init_dense(layer_t *l, int in_size, int out_size, activation_type_t act) {
    l->in_size  = in_size;
    l->out_size = out_size;
    l->activation = act;

    for(int o=0; o<out_size; o++) {
        l->biases[o] = 0;
        for(int i=0; i<in_size; i++) {
            // small random weights [-0.025,0.025]
            float w = (((float)rand() / RAND_MAX) - 0.5f) * 1.0f;
            l->weights[o][i] = fx_from_float(w);
        }
    }
}

void layer_forward(layer_t *l, const fx16_t *input) {
    for(int o=0;o<l->out_size;o++) {
        fx16_t acc = l->biases[o];
        for(int i=0;i<l->in_size;i++) {
            acc = fx_add(acc, fx_mul(l->weights[o][i], input[i]));
        }
        l->preact[o] = acc;
        l->output[o] = activation_forward(l->activation, acc);
    }
}

void layer_snapshot(layer_t *l) {
    for(int o=0;o<l->out_size;o++) {
        l->prev_output[o] = l->output[o];
        l->prev_preact[o] = l->preact[o];
    }
}

void layer_backward(layer_t *l, const fx16_t *input, fx16_t *grad_in, fx16_t lr) {
    for(int i=0;i<l->in_size;i++) grad_in[i] = 0;

    for(int o=0;o<l->out_size;o++) {
        fx16_t dact = activation_derivative(l->activation, l->output[o]);
        fx16_t delta = fx_mul(l->grad_out[o], dact);
        
        for(int i=0;i<l->in_size;i++) {
            fx16_t g = fx_mul(delta, input[i]);
            fx16_t update = fx_mul(lr, g);
            if(update > FX_FROM_FLOAT(0.1f)) update = FX_FROM_FLOAT(0.1f);
            if(update < FX_FROM_FLOAT(-0.1f)) update = FX_FROM_FLOAT(-0.1f);

            fx16_t w_old = l->weights[o][i];
            l->weights[o][i] = fx_sub(w_old, update);
            grad_in[i] = fx_add(grad_in[i], fx_mul(delta, w_old));
        }

        l->biases[o] = fx_sub(l->biases[o], fx_mul(lr, delta));
    }
}