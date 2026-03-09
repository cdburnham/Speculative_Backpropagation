#include "xor_json_common.h"
#include <stdio.h>
#include "nn_model_json.h"
#include "nn_train.h"

#define XOR_INPUTS 2
#define XOR_OUTPUTS 1
#define XOR_SAMPLES 4

static const nn_scalar_t XOR_X[XOR_SAMPLES * XOR_INPUTS] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

static const nn_scalar_t XOR_Y[XOR_SAMPLES * XOR_OUTPUTS] = {
    0,
    1,
    1,
    0
};

static nn_scalar_t xor_dataset_loss(nn_net_t* net, nn_loss_t loss) {
    nn_scalar_t out[XOR_OUTPUTS];
    nn_scalar_t acc = (nn_scalar_t)0;
    for (int i = 0; i < XOR_SAMPLES; i++) {
        nn_net_forward(net, &XOR_X[i * XOR_INPUTS], out);
        acc += nn_loss_forward(loss, out, &XOR_Y[i * XOR_OUTPUTS], XOR_OUTPUTS);
    }
    return acc / (nn_scalar_t)XOR_SAMPLES;
}

int xor_run_from_json(const char* json_path, int verbose, int* out_ok) {
    nn_model_config_t cfg;
    nn_net_t net;
    nn_scalar_t before_loss;
    nn_scalar_t after_loss;
    nn_scalar_t out[XOR_OUTPUTS];
    nn_status_t st;

    if (out_ok) *out_ok = 0;

    st = nn_model_config_load_json(json_path, &cfg);
    if (st != NN_OK) {
        printf("Failed to parse config: %s\n", json_path);
        return 1;
    }

    st = nn_model_build_from_config(&cfg, &net);
    if (st != NN_OK) {
        printf("Failed to build model: %s\n", json_path);
        return 1;
    }

    if (verbose) {
        printf("\n=== Config: %s ===\n", json_path);
        printf("Layers: %d  Input: %d  Output: %d\n",
               net.num_layers, net.input_dim, net.output_dim);
    }

    before_loss = xor_dataset_loss(&net, cfg.loss);
    printf("Initial loss: %.6f\n", (double)before_loss);

    if (cfg.use_spec1) {
        st = nn_train_sgd_spec1(&net, cfg.loss, XOR_X, XOR_Y, XOR_SAMPLES, cfg.epochs, &cfg.opt, cfg.print_every);
    } else {
        st = nn_train_sgd(&net, cfg.loss, XOR_X, XOR_Y, XOR_SAMPLES, cfg.epochs, &cfg.opt, cfg.print_every);
    }
    if (st != NN_OK) {
        printf("Training failed: %s\n", json_path);
        nn_net_free(&net);
        return 1;
    }

    after_loss = xor_dataset_loss(&net, cfg.loss);
    printf("Final loss:   %.6f\n", (double)after_loss);

    printf("Predictions:\n");
    for (int i = 0; i < XOR_SAMPLES; i++) {
        nn_net_forward(&net, &XOR_X[i * XOR_INPUTS], out);
        printf("  [%.0f %.0f] -> %.4f (target %.0f)\n",
               (double)XOR_X[i * XOR_INPUTS + 0],
               (double)XOR_X[i * XOR_INPUTS + 1],
               (double)out[0],
               (double)XOR_Y[i * XOR_OUTPUTS]);
    }

    if (out_ok) {
        *out_ok = (after_loss < before_loss) && (after_loss < (nn_scalar_t)0.02f);
    }

    nn_net_free(&net);
    return 0;
}
