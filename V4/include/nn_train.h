#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "nn_loss.h"
#include "nn_network.h"

#ifdef __cplusplus
extern "C" {
#endif

nn_status_t nn_train_sgd(nn_net_t* net,
                         nn_loss_t loss,
                         const nn_scalar_t* X,
                         const nn_scalar_t* Y,
                         int samples,
                         int epochs,
                         const nn_sgd_t* opt,
                         int print_every);

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
#endif

#endif
