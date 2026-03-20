## Speculative Backpropagation NN Library

## Built by: Cameron Burnham (camdburnham@icloud.com)
## Advised by: Arnab Purkayastha (arnab.purkayastha@wne.edu)
## Sponsored by: Ray Simar of Rice University (rs23@rice.edu)

## Git Status...

- V1 V2, and V3 have been archived due to deprecation and were removed from the active codebase.
- V4 is the active and supported implementation.
- Legacy reference documentation is available in `LEGACY_V1_V2_V3.md`.

This repository now focuses on V4 compatible with HLS.

## Defining V4...

V4 is a lightweight and static-memory C neural network library designed to be straightforward to synthesize/integrate in Vitis-oriented workflows:

- Allows modularized network synthesis on a variety of FPGA platforms supported by the Vitis/Vivado suite.
- Performance and limitations vary depending on a given FPGA device's resource utilization.
  - (note) Future versions might want to interface with a user's resource utilization reports for proper guardrails.
- Program-defined:
  - Math library for activation functions and derivative dependencies.
  - Network structure best defined from JSON files.
  - Dense (fully-connected) layers only
  - Activations: `linear`, `tanh`, `sigmoid`, `relu`
  - Loss: `mse`
  - Optimizer: plain SGD
  - Training modes:
    - Baseline SGD
    - `spec1` (1-step delayed speculative-style schedule)
- Static arrays in core NN path (Vitis-friendly)

## Repository directory structure...

- `V4/include/` public headers
- `V4/src/` core code and interfacing
- `V4/examples/` runnable examples and benchmarks
- `V4/data/models/` JSON model configs
- `V4/data/results/` CSV benchmark outputs

## Get familiar with the program's features with this workflow...

### 1. Build

```bash
cd V4
make clean
make
```

The above command produces the following binaries:

- `xor_train`
- `xor_json_from_file`
- `xor_json_suite`
- `xor_compare_csv`

### 2. Run a single model from a JSON configuration file.

```bash
./xor_json_from_file data/models/xor_small_tanh.json
```

The above command uses a pre-defined JSON configuration and example file to:

1. Parse the JSON config
2. Build/init the model
3. Train on XOR
4. Print initial/final loss and predictions

### 3. Run models in parallel and compare their benchmarks.

```bash
./xor_json_suite
```

The above command uses a predefined suite of example files to build/init, train (variable), and benchmark three different models:

- `xor_small_tanh.json`
- `xor_wide_tanh.json`
- `xor_deep_tanh.json`

### 4. Compare the baseline training mode (gradient descent) vs Spec1 mode (speculative backpropagation) and export the benchmarks to a CSV file for interpretation.

```bash
./xor_compare_csv data/models/xor_small_tanh.json data/results/xor_baseline_vs_spec.csv 1000
```

Execute the above command following the arguments provided below:

Arguments:

1. `model_path` (optional)
2. `csv_path` (optional)
3. `repeat_factor` (optional, integer > 0)

`repeat_factor` increases samples per epoch (`4 * repeat_factor`) for more stable timing.

Returned CSV columns:

- `epoch`
- `loss_baseline`
- `loss_spec`
- `time_baseline_ms`
- `time_spec_ms`

Note: `xor_compare_csv` caps epochs at 50,000 for limiting size. This can be modified directly in the scripting.

## Build your own model by defining a new JSON configuration...

### Reference:

Example: `V4/data/models/xor_small_tanh.json`

Supported keys:

- `max_layers` : int (<= compile-time cap)
- `max_width` : int (<= compile-time cap)
- `input_dim` : int
- `layers` : array of layer objects
  - `out_dim` : int
  - `activation` : `linear|tanh|sigmoid|relu`
- `init` : `uniform_sym|xavier|he`
- `seed` : int
- `loss` : currently `mse`
- `lr` : float
- `epochs` : int
- `print_every` : int
- `mode` : `sgd|spec1`

The lightweight example below defines a 2-input model limited to four layers at a maximum width of 16 neurons. It explicitly defines two layers (1. hidden: 4 neurons, tanh activation; 2. output: 1 neuron, normalized sigmoid). Weights are initialized using the Xavier method (sampled from a range defined by the layer's size and scaled by an equation defined in script). A random seed ensures weight initialization is a reproducible process across different training sessions. The loss function is MSE and utilizes a learning rate of 0.1 training at a maximum of 3000 epochs, printing its progress every 300 epochs. It uses spec1 mode to schedule training (more detail below).

```json
{
  "max_layers": 4,
  "max_width": 16,
  "input_dim": 2,
  "layers": [
    { "out_dim": 4, "activation": "tanh" },
    { "out_dim": 1, "activation": "sigmoid" }
  ],
  "init": "xavier",
  "seed": 123,
  "loss": "mse",
  "lr": 0.1,
  "epochs": 3000,
  "print_every": 300,
  "mode": "spec1"
}
```

### Guidelines:

Create a JSON configuration for a new model in `V4/data/models/`
(for example `my_model.json`).

Follow these guidelines to ensure you're working within the program's limits:

- Keep dimensions within `NN_CAP_MAX_LAYERS` and `NN_CAP_MAX_WIDTH`.
- Ensure each layer's `out_dim` is valid.
- Use stable `lr` first (start small, e.g. `0.01` to `0.1` depending on network depth).

### Step 1: Adjust your configuration file and train it:

```bash
./xor_json_from_file data/models/my_model.json
```

Validate it by confirming:

- Loss decreases as epochs increase.
- Final predictions align with supplied targets.

### Step 2: Compare SGD vs Spec1 Timing

```bash
./xor_compare_csv data/models/my_model.json data/results/my_model_compare.csv 5000
```

Compare and validate the two training modes against each other:

- Validate that both models' losses are finite and decreasing.
- Note the repeat factor affects time recording trends (to reduce timer noise).

### Step 3: Analyze CSV

My preference is Excel or RainbowCSV for quick CSV analysis:

The following figures are helpful for a basic analysis of performance and baseline comparison:

- Per-epoch speed ratio = `time_baseline_ms / time_spec_ms`
- Mean and median speed ratio across steady-state epochs
- Loss gap across modes

### Step 4: Refine and reiterate

Adjust your JSON configuration's values such as:

- Layer widths/depth
- Activation choices
- Learning rate
- Repeat factor for benchmarking

Re-run Steps 1-3 and identify changes in loss and timing trends.

## For those using Vitis...

- Core NN path avoids dynamic allocation in model execution/training loops.
- Static-capacity arrays bound memory use and make resource sizing explicit. These are adjustable and recompilable once you have an understanding of the program's influence on the board's resource utilization.
- Keep the code path simple before HLS optimization:
  - First verify correctness and convergence.
  - Only after verification and sanity checks should you introduce hardware-specific optimizations.

## Troubleshooting tips...

### Build fails?

- Run `make clean && make` in `V4`.
- Confirm you are using a C compiler with standard C support.

### Loss does not decrease?

- Lower `lr`
- Reduce model depth/width
- Verify output activation and loss pairing

### Timing looks noisy?

- Increase `repeat_factor`
- Compare aggregate/average times, not single-epoch outliers

## Current Recommendation...

Use V4 as the single source of truth for development, benchmarking, and any Vitis integration work.
