#ifndef SPECULATIVE_BP_H
#define SPECULATIVE_BP_H

#include "fx16.h"
#include "network.h"

// Speculative backprop — forward two candidate paths and commit best
// (pattern suitable for branch‑free hardware selection)

typedef struct {
    fx16_t temp_weights[MAX_PARAMS];
} spec_state_t;

void speculative_forward_commit(network_t *n, const fx16_t *input,
                                const fx16_t *target_a, const fx16_t *target_b,
                                fx16_t lr, int *choice_out);

#endif