#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void nn_act_forward(nn_activation_t act, nn_scalar_t* v, int n);
void nn_act_backward_mul(nn_activation_t act, const nn_scalar_t* y, nn_scalar_t* dy, int n);

#ifdef __cplusplus
}
#endif

#endif
