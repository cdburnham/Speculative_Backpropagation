#ifndef FX16_H
#define FX16_H

#include <stdint.h>
#include <math.h> // for tanhf, expf

typedef int16_t fx16_t;

#define FX_FRACTION_BITS 8

// ---------------------------
// Conversions
// ---------------------------
static inline fx16_t fx_from_float(float f) {
    return (fx16_t)(f * (1 << FX_FRACTION_BITS));
}

static inline float fx_to_float(fx16_t x) {
    return ((float)x) / (1 << FX_FRACTION_BITS);
}

static inline fx16_t fx_from_int(int x) {
    return (fx16_t)(x << FX_FRACTION_BITS);
}

// ---------------------------
// Arithmetic
// ---------------------------
#define fx_add(a,b) ((a)+(b))
#define fx_sub(a,b) ((a)-(b))

static inline fx16_t fx_mul(fx16_t a, fx16_t b) {
    return (fx16_t)((((int32_t)a * (int32_t)b) + (1 << (FX_FRACTION_BITS-1))) >> FX_FRACTION_BITS);
}

static inline fx16_t fx_div(fx16_t a, fx16_t b) {
    return (fx16_t)(((int32_t)a << FX_FRACTION_BITS) / b);
}

static inline fx16_t fx_abs(fx16_t x) {
    return (x < 0) ? -x : x;
}

static inline fx16_t fx_clamp(fx16_t x, fx16_t min, fx16_t max) {
    if(x < min) return min;
    if(x > max) return max;
    return x;
}

// ---------------------------
// Macros (must be before any function that uses them)
// ---------------------------
#define FX_FROM_FLOAT(f) fx_from_float(f)
#define FX_TO_FLOAT(x)   fx_to_float(x)
#define FX_ADD(a,b)      fx_add(a,b)
#define FX_SUB(a,b)      fx_sub(a,b)
#define FX_MUL(a,b)      fx_mul(a,b)
#define FX_DIV(a,b)      fx_div(a,b)
#define FX_ABS(x)        fx_abs(x)
#define FX_FROM_INT(i)   fx_from_int(i)
#define FX_CLAMP(x,mn,mx) fx_clamp(x,mn,mx)

// ---------------------------
// Activations
// ---------------------------

// ReLU
static inline fx16_t fx_relu(fx16_t x) {
    return (x < 0) ? 0 : x;
}
static inline fx16_t fx_relu_derivative(fx16_t y) {
    return (y > 0) ? fx_from_int(1) : 0; // direct call works now
}

// Tanh
static inline fx16_t fx_tanh(fx16_t x) {
    float f = fx_to_float(x);
    return fx_from_float(tanhf(f));
}
static inline fx16_t fx_tanh_derivative(fx16_t y) {
    return fx_sub(fx_from_int(1), fx_mul(y, y));
}

// Sigmoid
static inline fx16_t fx_sigmoid(fx16_t x) {
    float f = fx_to_float(x);
    float s = 1.0f / (1.0f + expf(-f));
    return fx_from_float(s);
}
static inline fx16_t fx_sigmoid_derivative(fx16_t y) {
    return fx_mul(y, fx_sub(fx_from_int(1), y));
}

#endif // FX16_H