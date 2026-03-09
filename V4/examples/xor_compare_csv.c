#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn_loss.h"
#include "nn_model_json.h"
#include "nn_network.h"

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

typedef struct {
    nn_scalar_t loss;
    double ms;
} epoch_metrics_t;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static inline const nn_scalar_t* sample_x(int idx) {
    return &XOR_X[(idx % XOR_SAMPLES) * XOR_INPUTS];
}

static inline const nn_scalar_t* sample_y(int idx) {
    return &XOR_Y[(idx % XOR_SAMPLES) * XOR_OUTPUTS];
}

static nn_status_t train_one_epoch_sgd(nn_net_t* net,
                                       nn_loss_t loss,
                                       const nn_sgd_t* opt,
                                       int train_samples,
                                       epoch_metrics_t* M) {
    nn_scalar_t pred[NN_CAP_MAX_WIDTH];
    nn_scalar_t acc = (nn_scalar_t)0;
    double t0 = now_ms();

    for (int n = 0; n < train_samples; n++) {
        nn_scalar_t sample_loss = (nn_scalar_t)0;
        nn_status_t st = nn_net_forward(net, sample_x(n), pred);
        if (st != NN_OK) return st;

        st = nn_net_backward_cached(net, loss, sample_y(n), &sample_loss);
        if (st != NN_OK) return st;

        nn_net_sgd_step(net, opt);
        acc += sample_loss;
    }

    M->loss = acc / (nn_scalar_t)train_samples;
    M->ms = now_ms() - t0;
    return NN_OK;
}

static nn_status_t train_one_epoch_spec1(nn_net_t* net,
                                         nn_loss_t loss,
                                         const nn_sgd_t* opt,
                                         int train_samples,
                                         epoch_metrics_t* M) {
    nn_scalar_t pred[NN_CAP_MAX_WIDTH];
    nn_scalar_t prev_inputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t prev_outputs[NN_CAP_MAX_LAYERS][NN_CAP_MAX_WIDTH] = {{0}};
    nn_scalar_t y_prev[NN_CAP_MAX_WIDTH] = {0};
    nn_scalar_t acc = (nn_scalar_t)0;
    double t0 = now_ms();

    {
        nn_status_t st = nn_net_forward(net, sample_x(0), pred);
        if (st != NN_OK) return st;

        for (int li = 0; li < net->num_layers; li++) {
            int in_dim = net->layers[li].in_dim;
            int out_dim = net->layers[li].out_dim;
            for (int i = 0; i < in_dim; i++) prev_inputs[li][i] = net->fwd_inputs[li][i];
            for (int i = 0; i < out_dim; i++) prev_outputs[li][i] = net->fwd_outputs[li][i];
        }
        for (int i = 0; i < net->output_dim; i++) y_prev[i] = sample_y(0)[i];
    }

    for (int n = 1; n < train_samples; n++) {
        nn_scalar_t sample_loss = (nn_scalar_t)0;
        nn_status_t st = nn_net_backward_with_cache(net, loss, prev_inputs, prev_outputs, y_prev, &sample_loss);
        if (st != NN_OK) return st;

        nn_net_sgd_step(net, opt);
        acc += sample_loss;

        st = nn_net_forward(net, sample_x(n), pred);
        if (st != NN_OK) return st;

        for (int li = 0; li < net->num_layers; li++) {
            int in_dim = net->layers[li].in_dim;
            int out_dim = net->layers[li].out_dim;
            for (int i = 0; i < in_dim; i++) prev_inputs[li][i] = net->fwd_inputs[li][i];
            for (int i = 0; i < out_dim; i++) prev_outputs[li][i] = net->fwd_outputs[li][i];
        }
        for (int i = 0; i < net->output_dim; i++) y_prev[i] = sample_y(n)[i];
    }

    {
        nn_scalar_t sample_loss = (nn_scalar_t)0;
        nn_status_t st = nn_net_backward_with_cache(net, loss, prev_inputs, prev_outputs, y_prev, &sample_loss);
        if (st != NN_OK) return st;

        nn_net_sgd_step(net, opt);
        acc += sample_loss;
    }

    M->loss = acc / (nn_scalar_t)train_samples;
    M->ms = now_ms() - t0;
    return NN_OK;
}

int main(int argc, char** argv) {
    const char* model_path = "data/models/xor_small_tanh.json";
    const char* csv_path = "data/results/xor_baseline_vs_spec.csv";
    int repeat_factor = 1;

    nn_model_config_t cfg;
    nn_net_t net_baseline;
    nn_net_t net_spec;
    FILE* fp;
    double total_baseline_ms = 0.0;
    double total_spec_ms = 0.0;
    int epochs_to_run;
    const int max_epochs = 50000;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) csv_path = argv[2];
    if (argc > 3) {
        repeat_factor = atoi(argv[3]);
        if (repeat_factor <= 0) repeat_factor = 1;
    }

    if (nn_model_config_load_json(model_path, &cfg) != NN_OK) {
        printf("Failed to load JSON config: %s\n", model_path);
        return 1;
    }

    if (nn_model_build_from_config(&cfg, &net_baseline) != NN_OK) {
        printf("Failed to build baseline model\n");
        return 1;
    }

    if (nn_model_build_from_config(&cfg, &net_spec) != NN_OK) {
        printf("Failed to build spec model\n");
        return 1;
    }

    fp = fopen(csv_path, "w");
    if (!fp) {
        printf("Failed to open CSV path: %s\n", csv_path);
        nn_net_free(&net_baseline);
        nn_net_free(&net_spec);
        return 1;
    }

    fprintf(fp, "epoch,loss_baseline,loss_spec,time_baseline_ms,time_spec_ms\n");

    epochs_to_run = cfg.epochs;
    if (epochs_to_run > max_epochs) {
        epochs_to_run = max_epochs;
        printf("Epochs capped at %d (config requested %d)\n", epochs_to_run, cfg.epochs);
    }

    printf("Model: %s\n", model_path);
    printf("CSV:   %s\n", csv_path);
    printf("Epochs: %d\n", epochs_to_run);
    printf("Train samples per epoch: %d\n", XOR_SAMPLES * repeat_factor);
    printf("Speculative mode: spec1 (minimal sequential reference)\n");

    for (int e = 1; e <= epochs_to_run; e++) {
        epoch_metrics_t mb;
        epoch_metrics_t ms;
        int train_samples = XOR_SAMPLES * repeat_factor;

        if (train_one_epoch_sgd(&net_baseline, cfg.loss, &cfg.opt, train_samples, &mb) != NN_OK) {
            printf("Baseline training failed at epoch %d\n", e);
            fclose(fp);
            nn_net_free(&net_baseline);
            nn_net_free(&net_spec);
            return 1;
        }

        if (train_one_epoch_spec1(&net_spec, cfg.loss, &cfg.opt, train_samples, &ms) != NN_OK) {
            printf("Speculative training failed at epoch %d\n", e);
            fclose(fp);
            nn_net_free(&net_baseline);
            nn_net_free(&net_spec);
            return 1;
        }

        total_baseline_ms += mb.ms;
        total_spec_ms += ms.ms;

        fprintf(fp, "%d,%.9f,%.9f,%.6f,%.6f\n",
                e, (double)mb.loss, (double)ms.loss, mb.ms, ms.ms);

        if (cfg.print_every > 0 && (e % cfg.print_every) == 0) {
            printf("Epoch %d  baseline=%.6f (%.3f ms)  spec=%.6f (%.3f ms)\n",
                   e, (double)mb.loss, mb.ms, (double)ms.loss, ms.ms);
        }
    }

    fclose(fp);
    nn_net_free(&net_baseline);
    nn_net_free(&net_spec);

    printf("Total baseline time: %.3f ms\n", total_baseline_ms);
    printf("Total spec time:     %.3f ms\n", total_spec_ms);
    if (total_spec_ms > 0.0) {
        printf("Observed speedup (baseline/spec): %.4fx\n", total_baseline_ms / total_spec_ms);
    }
    printf("Done. CSV written for plotting.\n");
    return 0;
}
