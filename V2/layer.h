#ifndef LAYER_H
#define LAYER_H

#include "fx16.h"
#include "config.h"
#include "activations.h"

typedef struct {
    int in_size, out_size;
    activation_type_t activation;

    fx16_t weights[MAX_NEURONS][MAX_NEURONS];
    fx16_t biases[MAX_NEURONS];

    fx16_t output[MAX_NEURONS];
    fx16_t preact[MAX_NEURONS];   // NEW

    fx16_t prev_output[MAX_NEURONS]; // NEW (spec)
    fx16_t prev_preact[MAX_NEURONS]; // NEW (spec)

    fx16_t grad_out[MAX_NEURONS];

    // Speculative gradient buffers
    fx16_t grad_w_spec[MAX_NEURONS][MAX_NEURONS];
    fx16_t grad_b_spec[MAX_NEURONS];
} layer_t;

void layer_init_dense(layer_t *l, int in_size, int out_size, activation_type_t act);
void layer_forward(layer_t *l, const fx16_t *input);
void layer_snapshot(layer_t *l);
void layer_backward(layer_t *l, const fx16_t *input, fx16_t *grad_in, fx16_t lr);

#endif