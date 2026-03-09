#include "nn_activation.h"
#include "nn_math.h"

void nn_act_forward(nn_activation_t act, nn_scalar_t* v, int n) {
    if (!v || n <= 0) return;

    for (int i = 0; i < n; i++) {
        nn_scalar_t x = v[i];
        switch (act) {
            case NN_ACT_LINEAR:
                break;
            case NN_ACT_TANH:
                v[i] = nn_tanh(x);
                break;
            case NN_ACT_SIGMOID:
                v[i] = nn_sigmoid(x);
                break;
            case NN_ACT_RELU:
                v[i] = x > (nn_scalar_t)0 ? x : (nn_scalar_t)0;
                break;
            default:
                break;
        }
    }
}

void nn_act_backward_mul(nn_activation_t act, const nn_scalar_t* y, nn_scalar_t* dy, int n) {
    if (!y || !dy || n <= 0) return;

    for (int i = 0; i < n; i++) {
        switch (act) {
            case NN_ACT_LINEAR:
                break;
            case NN_ACT_TANH:
                dy[i] *= ((nn_scalar_t)1 - y[i] * y[i]);
                break;
            case NN_ACT_SIGMOID:
                dy[i] *= y[i] * ((nn_scalar_t)1 - y[i]);
                break;
            case NN_ACT_RELU:
                dy[i] *= (y[i] > (nn_scalar_t)0) ? (nn_scalar_t)1 : (nn_scalar_t)0;
                break;
            default:
                break;
        }
    }
}
