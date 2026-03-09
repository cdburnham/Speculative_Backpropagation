#include "nn_rng.h"

uint32_t nn_rng_next(uint32_t* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

nn_scalar_t nn_rng_u01(uint32_t* state) {
    uint32_t r = nn_rng_next(state);
    return (nn_scalar_t)((r >> 8) & 0xFFFFFFu) / (nn_scalar_t)16777215.0f;
}

nn_scalar_t nn_rng_uniform_sym(uint32_t* state, nn_scalar_t scale) {
    return (nn_rng_u01(state) * (nn_scalar_t)2 - (nn_scalar_t)1) * scale;
}
