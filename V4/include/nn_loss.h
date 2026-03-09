#ifndef NN_LOSS_H // include guard
#define NN_LOSS_H

#include "nn_types.h" // include nn_scalar_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// PROTOTYPES:

// Computes the loss value given predictions and targets.
nn_scalar_t nn_loss_forward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n);

// Computes the gradient of the loss with respect to predictions.
void nn_loss_backward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n, nn_scalar_t* d_pred);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard