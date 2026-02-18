#include "fx16.h"
#include "memory.h"

unsigned layer_memory_bytes(const layer_t *l) {
    unsigned w = l->out_size * l->in_size * sizeof(fx16_t);
    unsigned b = l->out_size * sizeof(fx16_t);
    unsigned a = l->out_size * sizeof(fx16_t);
    return w + b + a;
}

unsigned network_memory_bytes(const network_t *n) {
    unsigned total = 0;
    for(int k=0;k<n->num_layers;k++) total += layer_memory_bytes(&n->layers[k]);
    return total;
}