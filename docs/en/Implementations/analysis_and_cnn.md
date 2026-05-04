---
title: English Analysis And CNN
nav_title: Analysis And CNN
parent: English Implementations
nav_order: 35
lang: en
---

# Analysis And CNN

## `analysis/BGK1D1V`

The analysis package provides benchmark, convergence, and cross-scheme utilities together with plotting helpers.

## `CNN/BGK1D1V`

The CNN code is currently organized into five layers.

- `util/`: model architecture, input-state encoding, and loss definitions
- `gen_traindata_v1/`: legacy NPZ-based data pipeline
- `gen_traindata_v2/`: shard-based PT pipeline with dataset and case manifests
- `train/`: training and sweep entry points
- `evaluation/`: cached warm-evaluation engine and compatibility wrappers

The current `train.py` also exposes `conv_type` and forwards it into the train-time warm-evaluation implicit configuration.

### Model and data conventions

- `MomentCNN1D` is a lightweight residual 1D CNN with an optional gated tail head.
- Inputs may be primitive (`nut`) or conservative (`nnuT`) and may optionally include temporal history via `prev_delta`.
- Targets are either `dw = (\Delta n, \Delta u, \Delta T)` or `dnu = (\Delta n, \Delta \nu, \Delta T)`.
- The implicit BGK1D stepper loads checkpoints and uses the predicted moment increments to initialize Picard iteration.
