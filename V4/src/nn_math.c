#include "nn_math.h"

nn_scalar_t nn_abs(nn_scalar_t x) {
    return x < (nn_scalar_t)0 ? -x : x;
}

nn_scalar_t nn_exp(nn_scalar_t x) {
    if (x > (nn_scalar_t)20) x = (nn_scalar_t)20;
    if (x < (nn_scalar_t)-20) x = (nn_scalar_t)-20;

    int k = 0;
    while (x > (nn_scalar_t)0.69314718f) {
        x -= (nn_scalar_t)0.69314718f;
        k++;
    }
    while (x < (nn_scalar_t)-0.69314718f) {
        x += (nn_scalar_t)0.69314718f;
        k--;
    }

    nn_scalar_t term = (nn_scalar_t)1;
    nn_scalar_t sum = (nn_scalar_t)1;
    for (int i = 1; i <= 8; i++) {
        term = term * x / (nn_scalar_t)i;
        sum += term;
    }

    while (k > 0) {
        sum *= (nn_scalar_t)2;
        k--;
    }
    while (k < 0) {
        sum *= (nn_scalar_t)0.5f;
        k++;
    }
    return sum;
}

nn_scalar_t nn_tanh(nn_scalar_t x) {
    if (x > (nn_scalar_t)4) return (nn_scalar_t)1;
    if (x < (nn_scalar_t)-4) return (nn_scalar_t)-1;

    nn_scalar_t ex = nn_exp(x);
    nn_scalar_t enx = nn_exp(-x);
    return (ex - enx) / (ex + enx);
}

nn_scalar_t nn_sigmoid(nn_scalar_t x) {
    if (x >= (nn_scalar_t)0) {
        nn_scalar_t z = nn_exp(-x);
        return (nn_scalar_t)1 / ((nn_scalar_t)1 + z);
    }
    nn_scalar_t z = nn_exp(x);
    return z / ((nn_scalar_t)1 + z);
}

nn_scalar_t nn_sqrt(nn_scalar_t x) {
    if (x <= (nn_scalar_t)0) return (nn_scalar_t)0;

    nn_scalar_t g = x > (nn_scalar_t)1 ? x : (nn_scalar_t)1;
    for (int i = 0; i < 8; i++) {
        g = (g + x / g) * (nn_scalar_t)0.5f;
    }
    return g;
}
