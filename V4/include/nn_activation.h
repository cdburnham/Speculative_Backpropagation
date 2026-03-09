#ifndef NN_ACTIVATION_H // include guard
#define NN_ACTIVATION_H

#include "nn_types.h" // import nn_status_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// PROTOTYPES:

// applies an activation function in-place to an array of values
void nn_act_forward(nn_activation_t act, nn_scalar_t* v, int n);
// multiplies the gradient by the derivative of the activation function
void nn_act_backward_mul(nn_activation_t act, const nn_scalar_t* y, nn_scalar_t* dy, int n);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard