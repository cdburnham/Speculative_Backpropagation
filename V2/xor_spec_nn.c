// fpga_nn_fx16_spec.c
// ------------------------------------------------------------
// Vivado HLS / Vitis friendly neural-net "library" in C:
//  - Static memory (no heap)
//  - Custom fx16 (int16 fixed-point) with saturating ops
//  - LUT sigmoid/tanh (precomputed at init, can become ROM/BRAM)
//  - Dense layers with per-layer activations
//  - Baseline SGD and 1-step speculative SGD (delayed update pipeline)
//  - Dataset abstraction (XOR now, BUTTER-E later)
//  - Demo: train 3 different nets sequentially
//
// CPU build/test:
//   gcc -O2 fpga_nn_fx16_spec.c -lm -o fpga_nn_fx16_spec
//
// For Vivado HLS:
//   - replace printf as needed
//   - add #pragma HLS PIPELINE / UNROLL on inner loops
//   - optionally move LUT init to host and store as const ROM
// ------------------------------------------------------------

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================
// Fixed-point config
// ============================================================
//
// Q4.12 fits well for NN values roughly in [-8, 8) with good fractional precision.
// If you expect larger pre-activations, reduce FRAC bits or add clamping.
#ifndef FX_FRAC_BITS
#define FX_FRAC_BITS 12
#endif

typedef int16_t fx16;

static inline int16_t sat16(int32_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

static inline fx16 fx_from_float(float f) {
    int32_t x = (int32_t)lroundf(f * (float)(1 << FX_FRAC_BITS));
    return (fx16)sat16(x);
}

static inline float fx_to_float(fx16 a) {
    return (float)a / (float)(1 << FX_FRAC_BITS);
}

static inline fx16 fx_add(fx16 a, fx16 b) {
    return (fx16)sat16((int32_t)a + (int32_t)b);
}

static inline fx16 fx_sub(fx16 a, fx16 b) {
    return (fx16)sat16((int32_t)a - (int32_t)b);
}

// Rounded fixed-point multiply: (a*b)>>FRAC with rounding
static inline fx16 fx_mul(fx16 a, fx16 b) {
    int32_t prod = (int32_t)a * (int32_t)b; // Q(2*frac)
    int32_t round = 1 << (FX_FRAC_BITS - 1);
    int32_t shifted = (prod + (prod >= 0 ? round : -round)) >> FX_FRAC_BITS;
    return (fx16)sat16(shifted);
}

// Accumulate dot-products in int32 Q(2*frac)
static inline void fx_mac_q2(int32_t *acc_q2, fx16 w, fx16 x) {
    *acc_q2 += (int32_t)w * (int32_t)x;
}

static inline fx16 fx_from_acc_q2(int32_t acc_q2) {
    int32_t round = 1 << (FX_FRAC_BITS - 1);
    int32_t shifted = (acc_q2 + (acc_q2 >= 0 ? round : -round)) >> FX_FRAC_BITS;
    return (fx16)sat16(shifted);
}

// Divide by small int (exact integer division on raw fx)
static inline fx16 fx_div_int(fx16 a, int d) {
    // (a / d) in Q(frac)
    return (fx16)(a / d);
}

// ============================================================
// LUT activations (avoid plateau vs crude PWL)
// ============================================================
//
// LUT maps x in [-ACT_XMAX, +ACT_XMAX] to sigmoid/tanh output.
// Outside that range, clamp.
//
// For FPGA: You can:
//  - generate LUT offline and store as const array
//  - OR keep init in software and push LUT into BRAM
//
#define ACT_LUT_SIZE 1024
#define ACT_XMAX     6.0f   // clamp range for LUT input

static fx16 sigmoid_lut[ACT_LUT_SIZE];
static fx16 tanh_lut[ACT_LUT_SIZE];

static inline int lut_index_from_x(fx16 x) {
    // x is Q(frac). Convert to float for init-time computations only is fine;
    // but at runtime we avoid float. We'll index using raw fixed.
    // Map x in [-XMAX, XMAX] to [0, LUT_SIZE-1].

    // Precompute fixed bounds:
    const fx16 xmin = fx_from_float(-ACT_XMAX);
    const fx16 xmax = fx_from_float(+ACT_XMAX);

    if (x <= xmin) return 0;
    if (x >= xmax) return ACT_LUT_SIZE - 1;

    // scale: (x - xmin) / (xmax - xmin) * (LUT_SIZE - 1)
    int32_t num = (int32_t)(x - xmin);
    int32_t den = (int32_t)(xmax - xmin);
    // num/den in raw fixed units, so do integer proportion:
    int32_t idx = (num * (ACT_LUT_SIZE - 1)) / den;
    if (idx < 0) idx = 0;
    if (idx > ACT_LUT_SIZE - 1) idx = ACT_LUT_SIZE - 1;
    return (int)idx;
}

static void act_lut_init(void) {
    for (int i = 0; i < ACT_LUT_SIZE; i++) {
        float t = (float)i / (float)(ACT_LUT_SIZE - 1);  // 0..1
        float x = -ACT_XMAX + t * (2.0f * ACT_XMAX);

        float s = 1.0f / (1.0f + expf(-x));
        float th = tanhf(x);

        sigmoid_lut[i] = fx_from_float(s);
        tanh_lut[i] = fx_from_float(th);
    }
}

typedef enum {
    ACT_LINEAR = 0,
    ACT_SIGMOID_LUT,
    ACT_TANH_LUT
} ActivationType;

static inline fx16 act_forward_fx(ActivationType t, fx16 x) {
    switch (t) {
        case ACT_LINEAR:
            return x;
        case ACT_SIGMOID_LUT:
            return sigmoid_lut[lut_index_from_x(x)];
        case ACT_TANH_LUT:
            return tanh_lut[lut_index_from_x(x)];
        default:
            return x;
    }
}

static inline fx16 act_deriv_from_a_fx(ActivationType t, fx16 a) {
    // derivative returned in Q(frac)
    const fx16 one = fx_from_float(1.0f);
    switch (t) {
        case ACT_LINEAR:
            return one;
        case ACT_SIGMOID_LUT:
            // a*(1-a)
            return fx_mul(a, fx_sub(one, a));
        case ACT_TANH_LUT:
            // 1-a^2
            return fx_sub(one, fx_mul(a, a));
        default:
            return one;
    }
}

// ============================================================
// Network limits (HLS-friendly)
// ============================================================

#define MAX_LAYERS   8
#define MAX_NEURONS  256   // max neurons per layer (incl hidden/output)
#define MAX_IN       256   // max fan-in for a layer

typedef struct {
    int in_dim;
    int out_dim;
    ActivationType act;

    // W[out_dim][in_dim] flattened with stride MAX_IN
    fx16 W[MAX_NEURONS * MAX_IN];
    fx16 b[MAX_NEURONS];
} DenseLayer;

typedef struct {
    int num_layers;
    fx16 lr;
    DenseLayer layers[MAX_LAYERS];
} Network;

// Cache: activations per stage (a[0]=input, a[l+1]=layer output)
typedef struct {
    fx16 a[MAX_LAYERS + 1][MAX_NEURONS];
} ActCache;

// Gradients
typedef struct {
    fx16 dW[MAX_LAYERS][MAX_NEURONS * MAX_IN];
    fx16 db[MAX_LAYERS][MAX_NEURONS];
} Grads;

// ============================================================
// Network API
// ============================================================

static void net_init(Network *net, fx16 lr) {
    memset(net, 0, sizeof(*net));
    net->lr = lr;
    net->num_layers = 0;
}

static int net_add_dense(Network *net, int in_dim, int out_dim, ActivationType act) {
    if (net->num_layers >= MAX_LAYERS) return -1;
    if (in_dim > MAX_IN || out_dim > MAX_NEURONS) return -2;

    DenseLayer *L = &net->layers[net->num_layers];
    L->in_dim = in_dim;
    L->out_dim = out_dim;
    L->act = act;

    // zero params
    memset(L->W, 0, sizeof(L->W));
    memset(L->b, 0, sizeof(L->b));

    net->num_layers++;
    return 0;
}

// Deterministic xorshift RNG (HLS-friendly)
static uint32_t rng_state = 1u;
static inline uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static inline float rand_f32_uniform(void) {
    // 24-bit mantissa-ish
    return (float)(xorshift32() & 0x00FFFFFF) / (float)0x01000000;
}

static void net_init_xavier(Network *net, uint32_t seed) {
    rng_state = seed ? seed : 1u;

    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *L = &net->layers[l];
        float limit = sqrtf(6.0f / (float)(L->in_dim + L->out_dim));

        for (int o = 0; o < L->out_dim; o++) {
            for (int i = 0; i < L->in_dim; i++) {
                float r = (rand_f32_uniform() * 2.0f - 1.0f) * limit;
                L->W[o * MAX_IN + i] = fx_from_float(r);
            }
            L->b[o] = fx_from_float(0.0f);
        }
    }
}

static void net_forward(const Network *net, const fx16 *input, ActCache *cache) {
    // copy input
    int in0 = net->layers[0].in_dim;
    for (int i = 0; i < in0; i++) cache->a[0][i] = input[i];

    for (int l = 0; l < net->num_layers; l++) {
        const DenseLayer *L = &net->layers[l];
        const fx16 *a_prev = cache->a[l];
        fx16 *a_out = cache->a[l + 1];

        for (int o = 0; o < L->out_dim; o++) {
            int32_t acc_q2 = 0;
            for (int i = 0; i < L->in_dim; i++) {
                fx_mac_q2(&acc_q2, L->W[o * MAX_IN + i], a_prev[i]);
            }
            // bias is Q(frac); acc is Q(2*frac) so lift bias:
            acc_q2 += ((int32_t)L->b[o]) << FX_FRAC_BITS;

            fx16 z = fx_from_acc_q2(acc_q2);
            a_out[o] = act_forward_fx(L->act, z);
        }
    }
}

static inline const fx16* net_output_ptr(const Network *net, const ActCache *cache) {
    return cache->a[net->num_layers];
}

static void grads_zero(Grads *g) {
    memset(g, 0, sizeof(*g));
}

// MSE loss for logging (fx)
static fx16 mse_fx(const fx16 *pred, const fx16 *target, int dim) {
    int32_t acc_q2 = 0;
    for (int i = 0; i < dim; i++) {
        fx16 d = fx_sub(pred[i], target[i]);
        acc_q2 += (int32_t)d * (int32_t)d; // Q(2*frac)
    }
    acc_q2 /= dim;
    return fx_from_acc_q2(acc_q2);
}

static void net_compute_grads_mse(const Network *net,
                                 const ActCache *cache,
                                 const fx16 *target,
                                 Grads *g)
{
    grads_zero(g);

    fx16 delta_next[MAX_NEURONS];
    fx16 delta_cur[MAX_NEURONS];
    memset(delta_next, 0, sizeof(delta_next));
    memset(delta_cur, 0, sizeof(delta_cur));

    int last = net->num_layers - 1;
    const DenseLayer *Llast = &net->layers[last];
    const fx16 *a_out = cache->a[last + 1];

    // scale = 2/out_dim (fixed)
    // Note: for XOR out_dim=1 so scale=2; for general tasks this matters.
    fx16 scale = fx_from_float(2.0f / (float)Llast->out_dim);

    for (int o = 0; o < Llast->out_dim; o++) {
        fx16 diff = fx_sub(a_out[o], target[o]);
        fx16 dL_da = fx_mul(scale, diff);
        fx16 da_dz = act_deriv_from_a_fx(Llast->act, a_out[o]);
        delta_next[o] = fx_mul(dL_da, da_dz);
    }

    // grads for last layer
    const fx16 *a_prev = cache->a[last];
    for (int o = 0; o < Llast->out_dim; o++) {
        g->db[last][o] = delta_next[o];
        for (int i = 0; i < Llast->in_dim; i++) {
            g->dW[last][o * MAX_IN + i] = fx_mul(delta_next[o], a_prev[i]);
        }
    }

    // hidden layers backward
    for (int l = last - 1; l >= 0; l--) {
        const DenseLayer *Cur = &net->layers[l];
        const DenseLayer *Next = &net->layers[l + 1];

        const fx16 *a_cur = cache->a[l + 1];
        const fx16 *a_prev2 = cache->a[l];

        for (int o = 0; o < Cur->out_dim; o++) {
            int32_t acc_q2 = 0;
            for (int k = 0; k < Next->out_dim; k++) {
                fx16 w = Next->W[k * MAX_IN + o]; // W_next[k][o]
                fx_mac_q2(&acc_q2, w, delta_next[k]);
            }
            fx16 sum = fx_from_acc_q2(acc_q2);
            fx16 da_dz = act_deriv_from_a_fx(Cur->act, a_cur[o]);
            delta_cur[o] = fx_mul(sum, da_dz);
        }

        for (int o = 0; o < Cur->out_dim; o++) {
            g->db[l][o] = delta_cur[o];
            for (int i = 0; i < Cur->in_dim; i++) {
                g->dW[l][o * MAX_IN + i] = fx_mul(delta_cur[o], a_prev2[i]);
            }
        }

        // shift
        memset(delta_next, 0, sizeof(delta_next));
        for (int o = 0; o < Cur->out_dim; o++) delta_next[o] = delta_cur[o];
        memset(delta_cur, 0, sizeof(delta_cur));
    }
}

static void net_apply_grads(Network *net, const Grads *g) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *L = &net->layers[l];
        for (int o = 0; o < L->out_dim; o++) {
            // b -= lr * db
            fx16 stepb = fx_mul(net->lr, g->db[l][o]);
            L->b[o] = fx_sub(L->b[o], stepb);

            for (int i = 0; i < L->in_dim; i++) {
                fx16 stepw = fx_mul(net->lr, g->dW[l][o * MAX_IN + i]);
                L->W[o * MAX_IN + i] = fx_sub(L->W[o * MAX_IN + i], stepw);
            }
        }
    }
}

// ============================================================
// Dataset abstraction (XOR now; BUTTER-E later)
// ============================================================

typedef struct {
    int n_samples;
    int in_dim;
    int out_dim;

    const fx16* (*get_x)(int idx, void *ctx);
    const fx16* (*get_y)(int idx, void *ctx);
    void *ctx;
} Dataset;

typedef struct {
    fx16 X[4][2];
    fx16 Y[4][1];
} XorData;

static const fx16* xor_get_x(int idx, void *ctx) {
    return ((XorData*)ctx)->X[idx];
}
static const fx16* xor_get_y(int idx, void *ctx) {
    return ((XorData*)ctx)->Y[idx];
}

static void xor_dataset_init(Dataset *ds, XorData *buf) {
    buf->X[0][0]=fx_from_float(0); buf->X[0][1]=fx_from_float(0);
    buf->X[1][0]=fx_from_float(0); buf->X[1][1]=fx_from_float(1);
    buf->X[2][0]=fx_from_float(1); buf->X[2][1]=fx_from_float(0);
    buf->X[3][0]=fx_from_float(1); buf->X[3][1]=fx_from_float(1);

    buf->Y[0][0]=fx_from_float(0);
    buf->Y[1][0]=fx_from_float(1);
    buf->Y[2][0]=fx_from_float(1);
    buf->Y[3][0]=fx_from_float(0);

    ds->n_samples = 4;
    ds->in_dim = 2;
    ds->out_dim = 1;
    ds->get_x = xor_get_x;
    ds->get_y = xor_get_y;
    ds->ctx = buf;
}

// ============================================================
// Training loops (baseline and speculative)
// ============================================================

static fx16 eval_epoch_loss(const Network *net, const Dataset *ds) {
    ActCache c;
    memset(&c, 0, sizeof(c));

    int32_t acc_q2 = 0;
    for (int n = 0; n < ds->n_samples; n++) {
        const fx16 *x = ds->get_x(n, ds->ctx);
        const fx16 *y = ds->get_y(n, ds->ctx);

        net_forward(net, x, &c);
        fx16 loss = mse_fx(net_output_ptr(net, &c), y, ds->out_dim);
        acc_q2 += ((int32_t)loss) << FX_FRAC_BITS;
    }
    acc_q2 /= ds->n_samples;
    return fx_from_acc_q2(acc_q2);
}

static void train_baseline_sgd(Network *net, const Dataset *ds, int epochs, int print_every) {
    ActCache c;
    memset(&c, 0, sizeof(c));
    Grads g;

    for (int e = 1; e <= epochs; e++) {
        for (int n = 0; n < ds->n_samples; n++) {
            const fx16 *x = ds->get_x(n, ds->ctx);
            const fx16 *y = ds->get_y(n, ds->ctx);

            net_forward(net, x, &c);
            net_compute_grads_mse(net, &c, y, &g);
            net_apply_grads(net, &g);
        }

        if (print_every > 0 && (e % print_every == 0 || e == 1 || e == epochs)) {
            fx16 loss = eval_epoch_loss(net, ds);
            printf("Epoch %d  Loss=%f\n", e, fx_to_float(loss));
        }
    }
}

static void train_speculative_sgd(Network *net, const Dataset *ds, int epochs, int print_every) {
    ActCache cur, prev;
    memset(&cur, 0, sizeof(cur));
    memset(&prev, 0, sizeof(prev));

    Grads g;
    fx16 prev_target[MAX_NEURONS];
    memset(prev_target, 0, sizeof(prev_target));

    int have_prev = 0;

    for (int e = 1; e <= epochs; e++) {
        have_prev = 0;

        for (int n = 0; n < ds->n_samples; n++) {
            const fx16 *x = ds->get_x(n, ds->ctx);
            const fx16 *y = ds->get_y(n, ds->ctx);

            // (A) forward current sample
            net_forward(net, x, &cur);

            // (B) update previous sample
            if (have_prev) {
                net_compute_grads_mse(net, &prev, prev_target, &g);
                net_apply_grads(net, &g);
            }

            // (C) shift current->prev
            prev = cur; // struct copy (static arrays)
            for (int i = 0; i < ds->out_dim; i++) prev_target[i] = y[i];
            have_prev = 1;
        }

        // flush last pending update
        if (have_prev) {
            net_compute_grads_mse(net, &prev, prev_target, &g);
            net_apply_grads(net, &g);
        }

        if (print_every > 0 && (e % print_every == 0 || e == 1 || e == epochs)) {
            fx16 loss = eval_epoch_loss(net, ds);
            printf("Epoch %d  Loss=%f\n", e, fx_to_float(loss));
        }
    }
}

// ============================================================
// Demo: build / train / print predictions
// ============================================================

static void print_xor_preds(const Network *net, const Dataset *ds, const char *title) {
    ActCache c;
    memset(&c, 0, sizeof(c));
    printf("\n%s\n", title);

    for (int n = 0; n < ds->n_samples; n++) {
        const fx16 *x = ds->get_x(n, ds->ctx);
        const fx16 *y = ds->get_y(n, ds->ctx);

        net_forward(net, x, &c);
        float p = fx_to_float(net_output_ptr(net, &c)[0]);
        printf("Input: %.0f %.0f  Pred: %.6f  True: %.0f\n",
               fx_to_float(x[0]), fx_to_float(x[1]),
               p, fx_to_float(y[0]));
    }
}

int main(void) {
    act_lut_init(); // LUTs for sigmoid/tanh

    Dataset ds;
    XorData xorbuf;
    xor_dataset_init(&ds, &xorbuf);

    // --------------------------------------------------------
    // Network 1: 2 -> 2 (tanh) -> 1 (sigmoid), speculative
    // --------------------------------------------------------
    printf("=== Network 1: 2-2-1, tanh/sigmoid, speculative ===\n");
    Network net1;
    net_init(&net1, fx_from_float(0.10f));
    net_add_dense(&net1, 2, 2, ACT_TANH_LUT);
    net_add_dense(&net1, 2, 1, ACT_SIGMOID_LUT);
    net_init_xavier(&net1, (uint32_t)time(NULL) ^ 0xA1u);

    train_speculative_sgd(&net1, &ds, /*epochs=*/20000, /*print_every=*/2000);
    print_xor_preds(&net1, &ds, "Net1 predictions:");

    // --------------------------------------------------------
    // Network 2: 2 -> 4 (tanh) -> 1 (sigmoid), baseline
    // --------------------------------------------------------
    printf("\n=== Network 2: 2-4-1, tanh/sigmoid, baseline ===\n");
    Network net2;
    net_init(&net2, fx_from_float(0.08f));
    net_add_dense(&net2, 2, 4, ACT_TANH_LUT);
    net_add_dense(&net2, 4, 1, ACT_SIGMOID_LUT);
    net_init_xavier(&net2, (uint32_t)time(NULL) ^ 0xB2u);

    train_baseline_sgd(&net2, &ds, /*epochs=*/15000, /*print_every=*/1500);
    print_xor_preds(&net2, &ds, "Net2 predictions:");

    // --------------------------------------------------------
    // Network 3: 2 -> 4 (tanh) -> 4 (tanh) -> 1 (sigmoid), speculative
    // --------------------------------------------------------
    printf("\n=== Network 3: 2-4-4-1, tanh/tanh/sigmoid, speculative ===\n");
    Network net3;
    net_init(&net3, fx_from_float(0.06f));
    net_add_dense(&net3, 2, 4, ACT_TANH_LUT);
    net_add_dense(&net3, 4, 4, ACT_TANH_LUT);
    net_add_dense(&net3, 4, 1, ACT_SIGMOID_LUT);
    net_init_xavier(&net3, (uint32_t)time(NULL) ^ 0xC3u);

    train_speculative_sgd(&net3, &ds, /*epochs=*/25000, /*print_every=*/2500);
    print_xor_preds(&net3, &ds, "Net3 predictions:");

    return 0;
}