# FPGA Neural Network C Library

# LEARN THE LIBRARY: Speculative Backpropagation + XOR Demo

This project implements specculative backpropagation in a lightweight, ANSI-C neural-network library designed to be friendly for FPGA HLS flows. It avoids dynamic allocation, and utilizes fixed-size buffers, and keeps control flow simple and predictable. A small demo implements tests a speculative backpropagation network on a XOR dataset and prints the calculated memory usage.

---

# Features

- Fixed-point arithmetic (Q8.8 by default)
- Dense layers with configurable activations
- Forward + backward pass suitable for HLS
- Speculative backprop pattern (evaluate two candidate outcomes, commit best)
- Global training knobs via a single macro block
- Layer-granular memory estimation
- Example XOR trainer for software test-benching

---

# Global Training Configuration

All training knobs are controlled from a single macro block in model demos:

```c
#define TRAINING_MAX_EPOCHS   5000
#define TRAINING_LEARNING_RATE fx_from_float(0.25f)
#define TRAINING_PRINT_EVERY   250