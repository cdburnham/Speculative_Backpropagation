#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "fx16.h"
#include "config.h"

typedef enum {
    ACT_LINEAR = 0,
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID
} activation_type_t;

fx16_t activation_forward(activation_type_t type, fx16_t x);
fx16_t activation_derivative(activation_type_t type, fx16_t x);

#endif