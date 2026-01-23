#ifndef NETWORK_H
#define NETWORK_H

#include "fx16.h"
#include "layer.h"

typedef struct {
    int num_layers;
    layer_t layers[MAX_LAYERS];
} network_t;

void network_init(network_t *n);
layer_t* network_add_dense(network_t *n, int in_size, int out_size, activation_type_t act);
void network_forward(network_t *n, const fx16_t *input);
void network_backward(network_t *n, const fx16_t *input, const fx16_t *target, fx16_t lr);

#endif