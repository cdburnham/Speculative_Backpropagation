#ifndef NN_MEMORY_H
#define NN_MEMORY_H

#include "nn_network.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t params_bytes;
    size_t grads_bytes;
    size_t workspace_bytes;
    size_t spec_cache_bytes;
    size_t total_bytes;
} nn_layer_mem_t;

typedef struct {
    nn_layer_mem_t* per_layer;
    size_t total_params;
    size_t total_grads;
    size_t total_workspace;
    size_t total_spec_cache;
    size_t total_all;
} nn_mem_report_t;

nn_status_t nn_net_estimate_memory(const nn_net_t* net, int include_spec_cache, nn_mem_report_t* report);
void nn_print_mem_report(const nn_net_t* net, const nn_mem_report_t* report);

#ifdef __cplusplus
}
#endif

#endif
