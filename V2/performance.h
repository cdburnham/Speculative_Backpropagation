#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include "fx16.h"
#include "config.h"

typedef struct {
    unsigned cycles_forward;
    unsigned cycles_backward;
    unsigned layers;
    unsigned params;
} perf_metrics_t;

void perf_reset(perf_metrics_t *p);
void perf_acc_forward(perf_metrics_t *p, unsigned c);
void perf_acc_backward(perf_metrics_t *p, unsigned c);

#endif