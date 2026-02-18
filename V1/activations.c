#include "fx16.h"
#include "activations.h"

fx16_t activation_forward(activation_type_t type, fx16_t x) {
#pragma HLS INLINE
    switch(type) {
        case ACT_RELU:    return (x > 0) ? x : 0;
        case ACT_TANH:    /* approx */ return x - fx_mul(x, fx_mul(x,x))/3;
        case ACT_SIGMOID: /* approx */ return fx_from_float(0.5f) + (x >> 2);
        default:          return x;
    }
}

fx16_t activation_derivative(activation_type_t type, fx16_t x) {
#pragma HLS INLINE
    switch(type) {
        case ACT_RELU:    return (x > 0) ? fx_from_float(1.0f) : 0;
        case ACT_TANH:    return fx_from_float(1.0f) - fx_mul(x,x);
        case ACT_SIGMOID: return fx_mul(x, fx_from_float(1.0f) - x);
        default:          return fx_from_float(1.0f);
    }
}