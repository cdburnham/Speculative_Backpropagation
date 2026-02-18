#include "fx16.h"
#include <stdio.h>
#include "network.h"
#include "memory.h"

#define TRAINING_MAX_EPOCHS    50000
#define TRAINING_LEARNING_RATE FX_FROM_FLOAT(0.08f)
#define TRAINING_PRINT_EVERY   5000

static const int XOR_INPUTS_INT[4][2] = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
};

static const int XOR_TARGETS_INT[4][1] = {
    {0},
    {1},
    {1},
    {0}
};

fx16_t XOR_INPUTS[4][2];
fx16_t XOR_TARGETS[4][1];

void convert_xor_dataset() {
    for(int i=0;i<4;i++){
        for(int j=0;j<2;j++)
            XOR_INPUTS[i][j] = fx_from_float((float)XOR_INPUTS_INT[i][j]);
            XOR_TARGETS[i][0] = fx_from_float((float)XOR_TARGETS_INT[i][0]);
    }
}

static network_t net;

// -----------------------------------------
// Build small XOR network
// -----------------------------------------
static void build_xor_network() {
    network_init(&net);
    network_add_dense(&net, 2, 4, ACT_TANH);
    network_add_dense(&net, 4, 1, ACT_TANH);
}

// -----------------------------------------
// Print memory usage per layer
// -----------------------------------------
static void print_memory_report() {
    printf("\n=== Memory Report ===\n");
    unsigned total = 0;

    for(int i=0; i<net.num_layers; i++){
        layer_t *l = &net.layers[i];
        unsigned bytes = layer_memory_bytes(l);
        total += bytes;
        printf("Layer %d: input=%d  output=%d  memory=%u bytes\n",
               i, l->in_size, l->out_size, bytes);
    }
    printf("Network total: %u bytes\n\n", total);
}

// -----------------------------------------
// Training loop
// -----------------------------------------
static void train_xor() {

    for(int epoch=0; epoch<TRAINING_MAX_EPOCHS; epoch++){
        for(int i=0; i<4; i++){
            network_forward(&net, XOR_INPUTS[i]);
            network_backward(&net,
                            XOR_INPUTS[i],
                            XOR_TARGETS[i],
                            TRAINING_LEARNING_RATE);
        }

        if(epoch % TRAINING_PRINT_EVERY == 0){
            fx16_t loss = 0;
            for(int i=0;i<4;i++){
                network_forward(&net, XOR_INPUTS[i]);
                fx16_t d = fx_sub(net.layers[net.num_layers-1].output[0], XOR_TARGETS[i][0]);
                loss = fx_add(loss, fx_mul(d, d));
            }
            printf("Epoch %d  Loss=%.4f\n", epoch, FX_TO_FLOAT(loss));
        }
    }
}

// -----------------------------------------
// Test network outputs
// -----------------------------------------
static void test_xor_outputs() {
    printf("\n=== XOR Output Test ===\n");
    for(int i=0;i<4;i++){
        network_forward(&net, XOR_INPUTS[i]);
        float y = FX_TO_FLOAT(net.layers[net.num_layers-1].output[0]);
        printf("%d XOR %d -> %.3f\n",
               (int)FX_TO_FLOAT(XOR_INPUTS[i][0]),
               (int)FX_TO_FLOAT(XOR_INPUTS[i][1]),
               y);
    }
}

// -----------------------------------------
// Main
// -----------------------------------------
int main() {
    convert_xor_dataset();
    build_xor_network();
    print_memory_report();
    train_xor();
    test_xor_outputs();
    return 0;
}