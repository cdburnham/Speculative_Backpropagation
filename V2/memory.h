#ifndef MEMORY_H
#define MEMORY_H

#include "fx16.h"
#include "network.h"

// Layer‑granular memory estimation (weights + biases + activations)

unsigned layer_memory_bytes(const layer_t *l);
unsigned network_memory_bytes(const network_t *n);

#endif