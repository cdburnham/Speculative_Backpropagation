#include "fx16.h"
#include "performance.h"

void perf_reset(perf_metrics_t *p) {
    p->cycles_forward = 0;
    p->cycles_backward = 0;
    p->layers = 0;
    p->params = 0;
}

void perf_acc_forward(perf_metrics_t *p, unsigned c) { p->cycles_forward += c; }
void perf_acc_backward(perf_metrics_t *p, unsigned c) { p->cycles_backward += c; }