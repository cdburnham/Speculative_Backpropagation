#ifndef NN_TRAIN_H // include guard
#define NN_TRAIN_H

#include "nn_loss.h" // include nn_loss_t and related functions
#include "nn_network.h" // include nn_net_t and related functions
#include "nn_types.h" // include nn_status_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// PROTOTYPES:

// Trains the neural network using stochastic gradient descent (SGD) for a specified number of epochs.
nn_status_t nn_train_sgd(nn_net_t* net,
                         nn_loss_t loss,
                         const nn_scalar_t* X,
                         const nn_scalar_t* Y,
                         int samples,
                         int epochs,
                         const nn_sgd_t* opt,
                         int print_every);

// Trains the neural network using a specific SGD variant that utilizes cached forward pass values for improved efficiency.
nn_status_t nn_train_sgd_spec1(nn_net_t* net,
                               nn_loss_t loss,
                               const nn_scalar_t* X,
                               const nn_scalar_t* Y,
                               int samples,
                               int epochs,
                               const nn_sgd_t* opt,
                               int print_every);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard