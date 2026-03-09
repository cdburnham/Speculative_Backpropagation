#ifndef NN_NETWORK_H // include guard
#define NN_NETWORK_H

#include "nn_layer_dense.h" // include nn_dense_t and related functions
#include "nn_types.h" // include nn_status_t and related types

#ifdef __cplusplus // declare C linkage for C++ compilers
extern "C" {
#endif

// Network structure and functions
typedef struct {
    int num_layers;
    int input_dim;
    int output_dim;

    nn_dense_t layers[NN_CAP_MAX_LAYERS];

    nn_scalar_t fwd_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH];
    nn_scalar_t fwd_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH];
} nn_net_t;

// PROTOTYPES:

// Creates an empty network structure. Must be called before adding layers or using the network.
nn_status_t nn_net_create(nn_net_t* net);

// Frees any resources associated with a network. Must be called when done using the network to avoid memory leaks.
void nn_net_free(nn_net_t* net);

// Adds a dense layer to the network with specified input dimension, output dimension, and activation function.
nn_status_t nn_net_add_dense(nn_net_t* net, int in_dim, int out_dim, nn_activation_t act);

// Initializes weights and biases of all layers in the network using the specified method and seed.
void nn_net_init(nn_net_t* net, nn_init_t init, uint32_t seed);

// Performs forward pass through the network, computing outputs from inputs.
nn_status_t nn_net_forward(nn_net_t* net, const nn_scalar_t* x, nn_scalar_t* out);

// Performs backward pass through the network, computing gradients and optionally returning the loss value.
nn_status_t nn_net_backward_cached(nn_net_t* net,
                                   nn_loss_t loss,
                                   const nn_scalar_t* target,
                                   nn_scalar_t* out_loss);

// Performs backward pass through the network using cached inputs and outputs from the forward pass, computing gradients and optionally returning the loss value.
nn_status_t nn_net_backward_with_cache(nn_net_t* net,
                                       nn_loss_t loss,
                                       const nn_scalar_t cached_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t cached_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH],
                                       const nn_scalar_t* target,
                                       nn_scalar_t* out_loss);

// Updates weights and biases of all layers in the network using SGD optimizer.
void nn_net_sgd_step(nn_net_t* net, const nn_sgd_t* opt);

#ifdef __cplusplus
}
#endif // end of C linkage

#endif // end of include guard