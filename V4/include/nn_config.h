#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#include "nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NN_CAP_MAX_LAYERS
#define NN_CAP_MAX_LAYERS 8
#endif

#ifndef NN_CAP_MAX_WIDTH
#define NN_CAP_MAX_WIDTH 64
#endif

extern int nn_g_max_layers;
extern int nn_g_max_width;

nn_status_t nn_set_global_constraints(int max_layers, int max_width);

#ifdef __cplusplus
}
#endif

#endif
