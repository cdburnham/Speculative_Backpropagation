#include "nn_config.h"

int nn_g_max_layers = NN_CAP_MAX_LAYERS;
int nn_g_max_width = NN_CAP_MAX_WIDTH;

nn_status_t nn_set_global_constraints(int max_layers, int max_width) {
    if (max_layers <= 0 || max_layers > NN_CAP_MAX_LAYERS) return NN_ERR_BADARG;
    if (max_width <= 0 || max_width > NN_CAP_MAX_WIDTH) return NN_ERR_BADARG;
    nn_g_max_layers = max_layers;
    nn_g_max_width = max_width;
    return NN_OK;
}
