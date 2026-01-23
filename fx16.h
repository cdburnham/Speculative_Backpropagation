#ifndef FX16_H
#define FX16_H

#include <stdint.h>

typedef int16_t fx16_t;

#define FX_FRACTION_BITS 8

#define fx_add(a,b) ((a)+(b))
#define fx_sub(a,b) ((a)-(b))

static inline fx16_t fx_from_float(float f) {
    return (fx16_t)(f * (1 << FX_FRACTION_BITS));
}

static inline float fx_to_float(fx16_t x) {
    return ((float)x) / (1 << FX_FRACTION_BITS);
}

// Fixed-point multiply
static inline fx16_t fx_mul(fx16_t a, fx16_t b) {
    return (fx16_t)((((int32_t)a * (int32_t)b) + (1 << (FX_FRACTION_BITS-1))) >> FX_FRACTION_BITS);
}

#define FX_FROM_FLOAT(f) fx_from_float(f)
#define FX_TO_FLOAT(x)   fx_to_float(x)
#define FX_MUL(a,b)      fx_mul(a,b)
#define FX_ADD(a,b)      fx_add(a,b)
#define FX_SUB(a,b)      fx_sub(a,b)

#endif // FX16_H