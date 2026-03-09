#ifndef NN_TYPES_H // include guard
#define NN_TYPES_H

#include <stddef.h> // include size_t
#include <stdint.h> // include uint32_t

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// Scalar type (Vitis-friendly: float). Switch to double if you want.
#ifndef NN_SCALAR_T
#define NN_SCALAR_T float
#endif

typedef NN_SCALAR_T nn_scalar_t;

// Status codes for function return values
typedef enum {
    NN_OK = 0,
    NN_ERR_BADARG = 1,
    NN_ERR_ALLOC = 2,
    NN_ERR_SHAPE = 3,
    NN_ERR_UNSUPPORTED = 4
} nn_status_t;

// Activation function types
typedef enum {
    NN_ACT_LINEAR = 0,
    NN_ACT_TANH   = 1,
    NN_ACT_SIGMOID= 2,
    NN_ACT_RELU   = 3
} nn_activation_t;

// Loss function types
typedef enum {
    NN_LOSS_MSE = 0
} nn_loss_t;

// Initialization types for weights and biases
typedef enum {
    NN_INIT_UNIFORM_SYM = 0, // Uniform distribution in the range [-scale, scale)
    NN_INIT_XAVIER      = 1, // Xavier/Glorot initialization
    NN_INIT_HE          = 2 // He initialization
} nn_init_t;

// Optimizer (SGD v1)
typedef struct {
    nn_scalar_t lr;
} nn_sgd_t;

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard