#include "nn_loss.h"

nn_scalar_t nn_loss_forward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n) {
    if (!pred || !target || n <= 0) return (nn_scalar_t)0;

    switch (loss) {
        case NN_LOSS_MSE: {
            nn_scalar_t acc = (nn_scalar_t)0;
            for (int i = 0; i < n; i++) {
                nn_scalar_t d = pred[i] - target[i];
                acc += d * d;
            }
            return acc / (nn_scalar_t)n;
        }
        default:
            return (nn_scalar_t)0;
    }
}

void nn_loss_backward(nn_loss_t loss, const nn_scalar_t* pred, const nn_scalar_t* target, int n, nn_scalar_t* d_pred) {
    if (!pred || !target || !d_pred || n <= 0) return;

    switch (loss) {
        case NN_LOSS_MSE:
            for (int i = 0; i < n; i++) {
                d_pred[i] = ((nn_scalar_t)2 * (pred[i] - target[i])) / (nn_scalar_t)n;
            }
            return;
        default:
            for (int i = 0; i < n; i++) {
                d_pred[i] = (nn_scalar_t)0;
            }
            return;
    }
}
