#ifndef NN_MATH_H
#define NN_MATH_H

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

nn_scalar_t nn_abs(nn_scalar_t x);
nn_scalar_t nn_exp(nn_scalar_t x);
nn_scalar_t nn_tanh(nn_scalar_t x);
nn_scalar_t nn_sigmoid(nn_scalar_t x);
nn_scalar_t nn_sqrt(nn_scalar_t x);

#ifdef __cplusplus
}
#endif

#endif
