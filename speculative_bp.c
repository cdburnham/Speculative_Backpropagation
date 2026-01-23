#include "fx16.h"
#include "speculative_bp.h"

static fx16_t mse_last_layer(layer_t *l, const fx16_t *target) {
    fx16_t e = 0;
    for(int o=0;o<l->out_size;o++) {
        fx16_t d = l->output[o] - target[o];
        e += fx_mul(d,d);
    }
    return e;
}

void speculative_forward_commit(network_t *n, const fx16_t *input,
                                const fx16_t *target_a, const fx16_t *target_b,
                                fx16_t lr, int *choice_out) {
    // Path A
    network_forward(n, input);
    fx16_t loss_a = mse_last_layer(&n->layers[n->num_layers-1], target_a);

    // Save snapshot (simple example — user may replace with dual‑buffer)
    // Path B reuse same weights but different loss target
    network_forward(n, input);
    fx16_t loss_b = mse_last_layer(&n->layers[n->num_layers-1], target_b);

    int commit_a = (loss_a <= loss_b);
    *choice_out = commit_a? 0:1;

    // Backprop only along chosen branch
    if(commit_a)
        network_backward(n, input, target_a, lr);
    else
        network_backward(n, input, target_b, lr);
}