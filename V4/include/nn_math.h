#ifndef NN_MATH_H // include guard
#define NN_MATH_H

#include "nn_types.h" // include nn_scalar_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// PROTOTYPES:

// Basic math functions. You can replace these with faster approximations if you want.
nn_scalar_t nn_abs(nn_scalar_t x);

// nn_pow is not in math.h for C17, so we provide our own simple implementation for positive integer exponents.
nn_scalar_t nn_exp(nn_scalar_t x);

// For simplicity, we only implement integer powers here. You can extend this to support fractional powers if you want.
nn_scalar_t nn_tanh(nn_scalar_t x);

// Sigmoid function and square root function.
nn_scalar_t nn_sigmoid(nn_scalar_t x);

// ReLU function and square root function.
nn_scalar_t nn_sqrt(nn_scalar_t x);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard