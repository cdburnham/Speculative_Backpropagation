#include "nn_memory.h"
#include <stdio.h>

static size_t bytes_params_dense(const nn_dense_t* L) {
    return ((size_t)L->out_dim * (size_t)L->in_dim + (size_t)L->out_dim) * sizeof(nn_scalar_t);
}

static size_t bytes_grads_dense(const nn_dense_t* L) {
    return ((size_t)L->out_dim * (size_t)L->in_dim + (size_t)L->out_dim) * sizeof(nn_scalar_t);
}

static size_t bytes_workspace_dense(const nn_dense_t* L) {
    return (size_t)L->out_dim * sizeof(nn_scalar_t);
}

static size_t bytes_spec_cache_dense(const nn_dense_t* L) {
    return ((size_t)L->in_dim + (size_t)L->out_dim) * sizeof(nn_scalar_t);
}

nn_status_t nn_net_estimate_memory(const nn_net_t* net, int include_spec_cache, nn_mem_report_t* report) {
    if (!net || !report || !report->per_layer) return NN_ERR_BADARG;

    report->total_params = 0;
    report->total_grads = 0;
    report->total_workspace = 0;
    report->total_spec_cache = 0;
    report->total_all = 0;

    for (int i = 0; i < net->num_layers; i++) {
        const nn_dense_t* L = &net->layers[i];
        nn_layer_mem_t* r = &report->per_layer[i];

        r->params_bytes = bytes_params_dense(L);
        r->grads_bytes = bytes_grads_dense(L);
        r->workspace_bytes = bytes_workspace_dense(L);
        r->spec_cache_bytes = include_spec_cache ? bytes_spec_cache_dense(L) : 0;
        r->total_bytes = r->params_bytes + r->grads_bytes + r->workspace_bytes + r->spec_cache_bytes;

        report->total_params += r->params_bytes;
        report->total_grads += r->grads_bytes;
        report->total_workspace += r->workspace_bytes;
        report->total_spec_cache += r->spec_cache_bytes;
        report->total_all += r->total_bytes;
    }

    return NN_OK;
}

void nn_print_mem_report(const nn_net_t* net, const nn_mem_report_t* report) {
    if (!net || !report || !report->per_layer) return;

    printf("\n=== NN V4 Memory Estimate (nn_scalar_t=%zu bytes) ===\n", sizeof(nn_scalar_t));
    printf("Global caps: layers=%d width=%d\n", nn_g_max_layers, nn_g_max_width);
    for (int i = 0; i < net->num_layers; i++) {
        const nn_dense_t* L = &net->layers[i];
        const nn_layer_mem_t* r = &report->per_layer[i];
        printf("Layer %d Dense %d->%d act=%d\n", i, L->in_dim, L->out_dim, (int)L->act);
        printf("  params:      %zu bytes\n", r->params_bytes);
        printf("  grads:       %zu bytes\n", r->grads_bytes);
        printf("  workspace:   %zu bytes\n", r->workspace_bytes);
        printf("  spec_cache:  %zu bytes\n", r->spec_cache_bytes);
        printf("  total:       %zu bytes\n", r->total_bytes);
    }
    printf("TOTAL params:     %zu bytes\n", report->total_params);
    printf("TOTAL grads:      %zu bytes\n", report->total_grads);
    printf("TOTAL workspace:  %zu bytes\n", report->total_workspace);
    printf("TOTAL spec_cache: %zu bytes\n", report->total_spec_cache);
    printf("TOTAL ALL:        %zu bytes\n", report->total_all);
    printf("===============================================\n\n");
}
