#ifndef NN_RNG_H
#define NN_RNG_H

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t nn_rng_next(uint32_t* state);
nn_scalar_t nn_rng_u01(uint32_t* state);
nn_scalar_t nn_rng_uniform_sym(uint32_t* state, nn_scalar_t scale);

#ifdef __cplusplus
}
#endif

#endif
