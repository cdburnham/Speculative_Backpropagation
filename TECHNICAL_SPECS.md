# V4 Technical Specifications

## 1. Scope

This document specifies the engineering behavior of the active `V4` implementation in this repository.

`V4` is a static-memory C neural-network library intended to be simple to reason about and practical for Vitis-oriented workflows.

Supported capabilities:

- Fully connected (dense) feed-forward networks
- Activations: `linear`, `tanh`, `sigmoid`, `relu`
- Loss: MSE
- Optimizer: SGD (`lr` only)
- Training schedules:
  - Standard SGD (`sgd`)
  - One-step delayed speculative schedule (`spec1`)

Non-goals in current version:

- Convolutional layers
- Batch training / mini-batch optimizers
- Regularization (dropout, weight decay, etc.)
- Advanced optimizers (Adam/RMSProp)
- Generic dataset loaders (examples are XOR-focused)

## 2. Build and Toolchain

Build system:

- `Makefile` in `V4/`
- Compiler defaults to `gcc`

Targets:

- `xor_train`
- `xor_json_from_file`
- `xor_json_suite`
- `xor_compare_csv`

Typical commands:

```bash
cd V4
make clean
make
```

No OpenMP or external runtime dependencies are required in the active baseline.

## 3. Memory Model and Capacity Constraints

### 3.1 Compile-time caps

Defined in `V4/include/nn_config.h`:

- `NN_CAP_MAX_LAYERS` (default: `8`)
- `NN_CAP_MAX_WIDTH` (default: `64`)

These control static array sizes across layers, caches, gradients, and temporary buffers.

### 3.2 Runtime constraints

Runtime limits are configured by `nn_set_global_constraints(max_layers, max_width)` and stored in globals:

- `nn_g_max_layers`
- `nn_g_max_width`

Model creation validates dims against these runtime constraints.

### 3.3 Allocation strategy

Core model representation is static (stack/struct arrays), not heap-allocated per layer.

- `nn_net_t` contains fixed arrays for layers and forward caches.
- `nn_dense_t` contains fixed arrays for params and grads.

This makes memory usage explicit and bounded for HLS-oriented reasoning.

## 4. Core Data Structures

## 4.1 Scalar and enums

From `V4/include/nn_types.h`:

- `nn_scalar_t` defaults to `float` (`NN_SCALAR_T` macro override supported)
- Status codes:
  - `NN_OK`
  - `NN_ERR_BADARG`
  - `NN_ERR_ALLOC`
  - `NN_ERR_SHAPE`
  - `NN_ERR_UNSUPPORTED`
- Activation enum: `LINEAR`, `TANH`, `SIGMOID`, `RELU`
- Loss enum: `MSE`
- Init enum: `UNIFORM_SYM`, `XAVIER`, `HE`
- Optimizer struct: `nn_sgd_t { nn_scalar_t lr; }`

## 4.2 Dense layer

`nn_dense_t` (`V4/include/nn_layer_dense.h`) contains:

- Dimensions and activation
- Parameters:
  - `W[out][in]`
  - `b[out]`
- Gradients:
  - `dW[out][in]`
  - `db[out]`
- Backprop scratch:
  - `delta[out]`

## 4.3 Network

`nn_net_t` (`V4/include/nn_network.h`) contains:

- Topology metadata: `num_layers`, `input_dim`, `output_dim`
- Layer array: `layers[NN_CAP_MAX_LAYERS]`
- Forward caches for each layer:
  - `fwd_inputs[layer][width]`
  - `fwd_outputs[layer][width]`

Forward caches are central for both standard and delayed (`spec1`) backward passes.

## 5. Computational Semantics

## 5.1 Forward pass

`nn_net_forward(net, x, out)`:

1. For each layer `li`:
   - Copy current input to `net->fwd_inputs[li]`
   - Run dense affine + activation into `net->fwd_outputs[li]`
2. Copy final layer output to user `out`

## 5.2 Backward pass with cache

`nn_net_backward_with_cache(...)`:

1. Compute loss and `d_pred` at network output.
2. Propagate gradients from last layer to first layer using cached inputs/outputs.
3. At each dense layer:
   - Build activation-adjusted delta
   - Compute `db`, `dW`
   - Compute gradient wrt previous layer input (`d_in`)

`nn_net_backward_cached(...)` is a convenience wrapper using `net->fwd_inputs/fwd_outputs` from most recent forward call.

## 5.3 Parameter update

`nn_net_sgd_step(net, opt)` applies:

- `b -= lr * db`
- `W -= lr * dW`

for every layer.

## 6. Initialization

Implemented in `nn_dense_init` (`V4/src/nn_layer_dense.c`):

- `uniform_sym`: scale `1`
- `xavier`: scale `sqrt(6 / (fan_in + fan_out))`
- `he`: scale `sqrt(6 / fan_in)`

Weights and biases are sampled via RNG utility (`nn_rng_uniform_sym`).

## 7. Training APIs

Defined in `V4/include/nn_train.h`.

## 7.1 `nn_train_sgd`

Per sample:

1. Forward with cache
2. Backward using current sample cache
3. SGD update

This is standard online SGD.

## 7.2 `nn_train_sgd_spec1`

Implements 1-step delay schedule:

1. Forward sample `t=0`, cache it
2. For each `t=1..N-1`:
   - Backward/update using previous sample cache (`t-1`)
   - Forward current sample `t`, cache it
3. Finalize by backward/update for last cached sample

Effect:

- Uses delayed cache for gradient step (speculative-style schedule)
- Same parameter update rule as SGD
- Primarily a scheduling variant, not a different optimizer

## 8. JSON Configuration Contract

Parser entry: `nn_model_config_load_json(path, cfg)`.

Builder entry: `nn_model_build_from_config(cfg, net)`.

Supported keys:

- `max_layers` (int)
- `max_width` (int)
- `input_dim` (int)
- `layers` (array)
  - `out_dim` (int)
  - `activation` (`linear|tanh|sigmoid|relu`)
- `init` (`uniform_sym|xavier|he`)
- `seed` (int)
- `loss` (`mse`)
- `lr` (float)
- `epochs` (int)
- `print_every` (int)
- `mode` (`sgd|spec1`)

Validation checks include:

- `max_layers` and `max_width` within compile-time caps
- positive `input_dim`, `epochs`
- positive layer count and valid per-layer output dims
- dimension chain consistency enforced when adding layers

Parser notes:

- Lightweight key search based parser (not full JSON grammar)
- File size cap: `NN_JSON_MAX_FILE_BYTES` (`16384`)

## 9. Example Programs

## 9.1 `xor_train`

- Loads default XOR model config
- Trains and prints periodic loss
- Prints final XOR predictions

## 9.2 `xor_json_from_file`

- Same as above but model path from CLI

## 9.3 `xor_json_suite`

- Runs multiple bundled model configs and reports pass/fail

## 9.4 `xor_compare_csv`

Purpose:

- Compare baseline SGD epoch metrics vs `spec1` epoch metrics on same architecture and hyperparameters.

Inputs:

1. model JSON path (optional)
2. CSV output path (optional)
3. repeat factor (optional)

Behavior:

- Trains two independent model instances initialized from same config
- Writes CSV columns:
  - `epoch`
  - `loss_baseline`
  - `loss_spec`
  - `time_baseline_ms`
  - `time_spec_ms`
- Caps epochs at `50000`

Timing implementation:

- Uses `clock_gettime(CLOCK_MONOTONIC)` and reports per-epoch wall time in milliseconds.

## 10. Error Handling Semantics

Most public APIs return `nn_status_t`.

Common failure classes:

- `NN_ERR_BADARG`: null pointer, invalid dimensions, invalid epoch/sample count
- `NN_ERR_SHAPE`: layer input dimension mismatch with previous layer output
- `NN_ERR_UNSUPPORTED`: cap overflow or unsupported runtime configuration

Resource lifecycle:

- `nn_net_create` zero-initializes struct
- `nn_net_free` zeroes struct (no dynamic free required in core net struct)

## 11. Determinism and Reproducibility

Weight initialization is seeded via JSON `seed`.

In `nn_net_init`, each layer seed is offset deterministically (`seed + i*101 + 7`) to avoid identical layer initialization while keeping repeatability.

## 12. Numerical and Performance Notes

- Scalar precision defaults to `float`.
- Very small workloads (e.g., XOR with 4 samples) produce noisy timing at sub-millisecond scale.
- For meaningful benchmark comparison, increase `repeat_factor` in `xor_compare_csv`.

## 13. Vitis Integration Guidance

Current design choices that support HLS migration:

- Fixed-size arrays for layers, params, caches, and grads
- Explicit loops and simple control flow in core math
- No dependency on external ML runtimes

Practical next HLS steps:

1. Fix target dimensions/caps for your accelerator SKU.
2. Isolate hot loops (`nn_dense_forward`, `nn_dense_backward`, SGD step).
3. Add pragmas/pipelining directives in a Vitis branch while preserving software reference behavior.
4. Validate numerical drift vs software baseline with shared test vectors.

## 14. Known Limitations

- Single-sample SGD only (no batching)
- Only MSE loss
- Simple parser (expects expected key/value formatting)
- Dense-only architecture
- No built-in checkpointing / serialization of trained weights

## 15. File Map (Active)

- Public interfaces:
  - `V4/include/nn_types.h`
  - `V4/include/nn_config.h`
  - `V4/include/nn_layer_dense.h`
  - `V4/include/nn_network.h`
  - `V4/include/nn_loss.h`
  - `V4/include/nn_activation.h`
  - `V4/include/nn_train.h`
  - `V4/include/nn_model_json.h`
- Core implementation:
  - `V4/src/nn_layer_dense.c`
  - `V4/src/nn_network.c`
  - `V4/src/nn_loss.c`
  - `V4/src/nn_activation.c`
  - `V4/src/nn_train.c`
  - `V4/src/nn_model_json.c`
  - plus utilities in `V4/src/nn_math.c`, `V4/src/nn_rng.c`, `V4/src/nn_config.c`
- Executables / workflows:
  - `V4/examples/xor_train.c`
  - `V4/examples/xor_json_from_file.c`
  - `V4/examples/xor_json_suite.c`
  - `V4/examples/xor_compare_csv.c`
