#include "fx16.h"
#include "network.h"
#include "layer.h"
#include <stddef.h>

void network_init(network_t *n) {
    n->num_layers = 0;
}

layer_t* network_add_dense(network_t *n, int in_size, int out_size, activation_type_t act) {
    if(n->num_layers >= MAX_LAYERS) return NULL;
    layer_t *l = &n->layers[n->num_layers++];
    layer_init_dense(l, in_size, out_size, act);
    return l;
}

void network_forward(network_t *n, const fx16_t *input) {
    const fx16_t *inp = input;
    for(int i=0; i<n->num_layers; i++) {
        layer_forward(&n->layers[i], inp);
        inp = n->layers[i].output;
    }
}

void network_backward(network_t *n, const fx16_t *input, const fx16_t *target, fx16_t lr) {
    fx16_t grad[MAX_NEURONS];
    fx16_t *next_grad = grad;

    layer_t *last = &n->layers[n->num_layers-1];
    for(int o=0;o<last->out_size;o++)
        last->grad_out[o] = fx_sub(last->output[o], target[o]);

    for(int i=n->num_layers-1;i>=0;i--) {
        const fx16_t *inp = (i==0) ? input : n->layers[i-1].output;
        layer_backward(&n->layers[i], inp, next_grad, lr);
        next_grad = grad;
    }
}