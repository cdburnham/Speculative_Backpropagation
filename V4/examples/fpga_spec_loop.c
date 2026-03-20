#include <stdio.h>
#include <stdlib.h>

#include "nn_loss.h"
#include "nn_network.h"
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

int main(int argc, char** argv) {
    int epochs_per_block = 1000;
    int print_every = 250;
    nn_scalar_t lr = (nn_scalar_t)0.1f;
    uint32_t seed = 123u;

    nn_net_t net;
    nn_sgd_t opt;
    nn_status_t st;
    long long total_epochs = 0;
    long long block_idx = 0;

    if (argc > 1) epochs_per_block = atoi(argv[1]);
    if (argc > 2) print_every = atoi(argv[2]);

    if (epochs_per_block <= 0) {
        printf("Invalid epochs_per_block: %d\n", epochs_per_block);
        return 1;
    }
    if (print_every <= 0) {
        print_every = epochs_per_block;
    }

    st = nn_net_create(&net);
    if (st != NN_OK) {
        printf("Failed to create network\n");
        return 1;
    }

    st = nn_net_add_dense(&net, 2, 4, NN_ACT_TANH);
    if (st != NN_OK) {
        printf("Failed to add hidden layer\n");
        nn_net_free(&net);
        return 1;
    }
    st = nn_net_add_dense(&net, 4, 1, NN_ACT_SIGMOID);
    if (st != NN_OK) {
        printf("Failed to add output layer\n");
        nn_net_free(&net);
        return 1;
    }
    nn_net_init(&net, NN_INIT_XAVIER, seed);

    opt.lr = lr;

    printf("Starting speculative training loop (infinite).\n");
    printf("Model: inline 2-4-1 (tanh -> sigmoid)\n");
    printf("Mode: spec1\n");
    printf("Epochs per block: %d\n", epochs_per_block);
    printf("Print every: %d\n", print_every);
    printf("Learning rate: %.6f\n", (double)opt.lr);
    printf("Stop by terminating the app, halting the target, or resetting the board.\n\n");

    while (1) {
        nn_scalar_t loss;
        nn_scalar_t out[XOR_OUTPUTS];

        block_idx++;
        st = nn_train_sgd_spec1(
            &net,
            NN_LOSS_MSE,
            XOR_X,
            XOR_Y,
            XOR_SAMPLES,
            epochs_per_block,
            &opt,
            print_every
        );
        if (st != NN_OK) {
            printf("Training failed at block %lld\n", block_idx);
            nn_net_free(&net);
            return 2;
        }

        total_epochs += epochs_per_block;
        loss = xor_dataset_loss(&net, NN_LOSS_MSE);

        printf("Block %lld complete. Total epochs: %lld  Loss: %.6f\n",
               block_idx, total_epochs, (double)loss);

        for (int i = 0; i < XOR_SAMPLES; i++) {
            nn_net_forward(&net, &XOR_X[i * XOR_INPUTS], out);
            printf("  [%.0f %.0f] -> %.4f (target %.0f)\n",
                   (double)XOR_X[i * XOR_INPUTS + 0],
                   (double)XOR_X[i * XOR_INPUTS + 1],
                   (double)out[0],
                   (double)XOR_Y[i * XOR_OUTPUTS]);
        }
        printf("\n");
    }

    return 0;
}
