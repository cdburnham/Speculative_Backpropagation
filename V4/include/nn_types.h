#ifndef NN_TYPES_H
#define NN_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scalar type (Vitis-friendly: float). Switch to double if you want.
#ifndef NN_SCALAR_T
#define NN_SCALAR_T float
#endif

typedef NN_SCALAR_T nn_scalar_t;

// Status codes
typedef enum {
    NN_OK = 0,
    NN_ERR_BADARG = 1,
    NN_ERR_ALLOC = 2,
    NN_ERR_SHAPE = 3,
    NN_ERR_UNSUPPORTED = 4
} nn_status_t;

// Activation types
typedef enum {
    NN_ACT_LINEAR = 0,
    NN_ACT_TANH   = 1,
    NN_ACT_SIGMOID= 2,
    NN_ACT_RELU   = 3
} nn_activation_t;

// Loss types
typedef enum {
    NN_LOSS_MSE = 0
} nn_loss_t;

// Init types
typedef enum {
    NN_INIT_UNIFORM_SYM = 0, // uniform in [-scale, +scale]
    NN_INIT_XAVIER      = 1,
    NN_INIT_HE          = 2
} nn_init_t;

// Optimizer (SGD v1)
typedef struct {
    nn_scalar_t lr;
} nn_sgd_t;

#ifdef __cplusplus
}
#endif

#endif
