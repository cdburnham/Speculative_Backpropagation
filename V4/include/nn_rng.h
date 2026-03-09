#ifndef NN_RNG_H // include guard
#define NN_RNG_H

#include "nn_types.h" // include nn_scalar_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// PROTOTYPES:

// Generates the next random number in the sequence and updates the state.
uint32_t nn_rng_next(uint32_t* state);

// Generates a random number uniformly distributed in the range [0, 1).
nn_scalar_t nn_rng_u01(uint32_t* state);

// Generates a random number uniformly distributed in the range [-scale, scale).
nn_scalar_t nn_rng_uniform_sym(uint32_t* state, nn_scalar_t scale);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard