# Legacy Documentation: V1, V2, V3

This document preserves engineering notes for the archived `V1`, `V2`, and `V3` phases.

These versions are deprecated and removed from the active tree. The supported implementation is `V4`.

## Status Summary

- `V1`: archived
- `V2`: archived
- `V3`: archived (implemented as the third iteration inside `V2`, not as a separate `V3/` folder)
- Active version: `V4`

## Where Legacy Code Lives

Legacy code can be checked out from git history:

- Last commit containing `V1/` and `V2/`: `99dc62e`
- Transition to `V4`-focused work: `dd40395`

Example commands:

```bash
git checkout 99dc62e
# or inspect files without checkout
git show 99dc62e:V1/README.md
git show 99dc62e:V2/README.md
```

## V1 Technical Snapshot

## Purpose

- First structured static-memory NN library for FPGA-oriented C experimentation.
- Introduced speculative backpropagation as a selectable training behavior.

## Folder Contents (historical)

- `V1/README.md`
- `V1/config.h`
- `V1/fx16.h`
- `V1/network.c`, `V1/network.h`
- `V1/layer.c`, `V1/layer.h`
- `V1/activations.c`, `V1/activations.h`
- `V1/speculative_bp.c`, `V1/speculative_bp.h`
- `V1/performance.c`, `V1/performance.h`
- `V1/memory.c`, `V1/memory.h`
- `V1/xor_demo.c`
- `V1/build.sh`

## Key Characteristics

- Fixed-point path (`fx16`) with static buffers.
- Dense feedforward network only.
- XOR-based training/demo flow.
- Manual shell-based build workflow (`build.sh` + `clang`).
- Memory reporting utilities included.

## V2 Technical Snapshot

## Purpose

- Expanded experimentation branch after V1.
- Added multiple XOR executables for baseline and speculative variants.

## Folder Contents (historical)

- All major V1-style modules remained (`network`, `layer`, `activations`, `speculative_bp`, `memory`, `performance`, `fx16`, `config`).
- Additional demos:
  - `V2/xor_demo.c`
  - `V2/xor_demo_v2.c`
  - `V2/xor_demo_v3.c`
  - `V2/xor_spec_nn.c`
- Build helper: `V2/build.sh`

## Key Characteristics

- Continued static-memory C style.
- Side-by-side comparison of training logic across demo variants.
- Larger experimental file (`xor_spec_nn.c`) with richer comments and algorithm variants.
- Still script-driven local compilation (not a unified library-style build system).

## V3 Technical Snapshot

## Clarification

`V3` was developed as a phase/iteration and represented by files such as:

- `V2/xor_demo_v3.c`
- related speculative pipeline work in `V2/xor_spec_nn.c`

There was no standalone top-level `V3/` directory in repository history.

## Purpose

- Third-pass refinement of XOR training behavior and speculative/backprop comparison.
- Stepping stone toward the simplified, consolidated `V4` architecture.

## Why V1–V3 Were Deprecated

- Inconsistent interfaces across demos and phases.
- Experimental code paths made maintenance and onboarding harder.
- Build/run workflow was less uniform than desired for Vitis-centric use.
- `V4` unified configuration, API shape, examples, and documentation into one maintained path.

## Migration Guidance

If you are reading legacy results/scripts:

1. Treat V1–V3 as reference-only historical prototypes.
2. Port ideas into `V4` modules instead of reviving old trees.
3. Use `V4/data/models/*.json` and `V4/examples/*` for all current benchmarking and board preparation work.

