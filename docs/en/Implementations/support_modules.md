---
title: English Support Modules
nav_title: Support Modules
parent: English Implementations
grand_parent: English
nav_order: 37
lang: en
---

# Support Modules

## `params/`

- defines model dataclasses for BGK1D and BGK2D2V
- provides default scheme-parameter factories
- supplies the model-config type information used by the API layer

## `utillib/`

- `device_util.py`: validates `cuda`, `mps`, and `cpu`
- `progress_bar.py`: wraps optional tqdm usage
- `pretty.py`: formats dataclasses and dicts for logging

## `plotting/`

- `plotting/bgk1d/plot_state.py` visualizes `f`, `n`, `u`, and `T`

## `tests/`

The test tree contains smoke checks and utility-level validation. Many tests are environment-sensitive and may be skipped in non-CUDA settings.
