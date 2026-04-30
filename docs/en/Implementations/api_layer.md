---
title: English API Layer
nav_title: API Layer
parent: English Implementations
grand_parent: English
nav_order: 32
lang: en
---

# API Layer

## Target directory

- `src/kineticEQ/api/`

## File responsibilities

### `config.py`

- enum definitions for models, schemes, backends, dtypes, and logging
- alias-aware `parse_enum()`
- `Config` dataclass and normalization logic
- convenience accessors such as `as_dict`

### `engine.py`

- fills and validates `model_cfg`
- fills default `scheme_params`
- applies logging and validates devices
- builds the state and the stepper
- executes the runtime loop with progress reporting

### `result.py`

- `Result(metrics, payload)` dataclass

### `logging_utils.py`

- consistent logger setup for the `kineticEQ` namespace

## Key implementation note

`Engine` assumes that `config.model_cfg.scheme_params` exists. This is why BGK2D2V currently fails in the Engine path.
