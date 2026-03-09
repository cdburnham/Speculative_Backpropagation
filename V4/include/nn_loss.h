#ifndef NN_LOSS_H
#define NN_LOSS_H

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

nn_scalar_t nn_loss_forward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n);
void nn_loss_backward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n, nn_scalar_t* d_pred);

#ifdef __cplusplus
}
#endif

#endif
