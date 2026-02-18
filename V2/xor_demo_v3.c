#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// XOR dataset
#define INPUTS 2
#define HIDDEN 2
#define OUTPUTS 1
#define SAMPLES 4
#define EPOCHS 100000000
#define LR 0.1

double X[SAMPLES][INPUTS] = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
};

double Y[SAMPLES][OUTPUTS] = {
    {0},
    {1},
    {1},
    {0}
};

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double y) {
    return y * (1.0 - y); // derivative assuming y = sigmoid(x)
}

double tanh_deriv(double y) {
    return 1.0 - y * y; // derivative assuming y = tanh(x)
}

// Initialize weights
double w1[INPUTS][HIDDEN];
double b1[HIDDEN];
double w2[HIDDEN][OUTPUTS];
double b2[OUTPUTS];

double hidden[HIDDEN];
double output[OUTPUTS];

void init_weights() {
    srand(time(NULL));
    for(int i=0;i<INPUTS;i++)
        for(int j=0;j<HIDDEN;j++)
            w1[i][j] = ((double)rand()/RAND_MAX)*2 - 1; // -1 to 1
    for(int j=0;j<HIDDEN;j++)
        b1[j] = ((double)rand()/RAND_MAX)*2 - 1;
    for(int j=0;j<HIDDEN;j++)
        for(int k=0;k<OUTPUTS;k++)
            w2[j][k] = ((double)rand()/RAND_MAX)*2 - 1;
    for(int k=0;k<OUTPUTS;k++)
        b2[k] = ((double)rand()/RAND_MAX)*2 - 1;
}

// Forward pass
void forward(double x[INPUTS]) {
    for(int j=0;j<HIDDEN;j++){
        hidden[j] = b1[j];
        for(int i=0;i<INPUTS;i++)
            hidden[j] += x[i]*w1[i][j];
        hidden[j] = tanh(hidden[j]);
    }
    for(int k=0;k<OUTPUTS;k++){
        output[k] = b2[k];
        for(int j=0;j<HIDDEN;j++)
            output[k] += hidden[j]*w2[j][k];
        output[k] = sigmoid(output[k]);
    }
}

// Backward pass and update weights
void backward(double x[INPUTS], double y[OUTPUTS]) {
    double delta_out[OUTPUTS];
    for(int k=0;k<OUTPUTS;k++)
        delta_out[k] = (output[k] - y[k]) * sigmoid_deriv(output[k]);

    double delta_hidden[HIDDEN];
    for(int j=0;j<HIDDEN;j++){
        double sum = 0;
        for(int k=0;k<OUTPUTS;k++)
            sum += delta_out[k]*w2[j][k];
        delta_hidden[j] = sum * tanh_deriv(hidden[j]);
    }

    // Update weights and biases
    for(int j=0;j<HIDDEN;j++){
        for(int k=0;k<OUTPUTS;k++)
            w2[j][k] -= LR * delta_out[k] * hidden[j];
    }
    for(int k=0;k<OUTPUTS;k++)
        b2[k] -= LR * delta_out[k];

    for(int i=0;i<INPUTS;i++){
        for(int j=0;j<HIDDEN;j++)
            w1[i][j] -= LR * delta_hidden[j] * x[i];
    }
    for(int j=0;j<HIDDEN;j++)
        b1[j] -= LR * delta_hidden[j];
}

// Mean Squared Error
double mse() {
    double loss = 0;
    for(int n=0;n<SAMPLES;n++){
        forward(X[n]);
        for(int k=0;k<OUTPUTS;k++){
            double diff = output[k] - Y[n][k];
            loss += diff*diff;
        }
    }
    return loss / SAMPLES;
}

int main() {
    init_weights();

    for(int epoch=1;epoch<=EPOCHS;epoch++){
        for(int n=0;n<SAMPLES;n++){
            forward(X[n]);
            backward(X[n], Y[n]);
        }
        double loss = mse();
        printf("Epoch %d  Loss=%.6f\n", epoch, loss);
    }

    // Print final predictions
    printf("\nFinal predictions:\n");
    for(int n=0;n<SAMPLES;n++){
        forward(X[n]);
        printf("Input: %.0f %.0f  Predicted: %.4f  True: %.0f\n",
            X[n][0], X[n][1], output[0], Y[n][0]);
    }

    return 0;
}