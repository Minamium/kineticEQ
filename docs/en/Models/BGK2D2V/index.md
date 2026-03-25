---
title: English BGK2D2V
nav_title: BGK2D2V
parent: English Models
nav_order: 22
has_children: true
lang: en
---

# BGK2D2V

BGK2D2V denotes the two-dimensional-in-space, two-dimensional-in-velocity BGK model. The current repository contains state allocation code and parameter dataclasses, but the full Engine path is not operational.

## Current status

1. `State2D2V` allocates `f`, `f_new`, `x`, `y`, `vx`, and `vy`.
2. `bgk2d2v_explicit_torch.py` is registered, but `step()` is still a TODO placeholder.
3. `BGK2D2V.ModelConfig` does not contain `scheme_params`, which breaks the current `Engine` initialization path.
4. A CUDA implementation exists under `cuda_kernel/BGK2D2V/explicit_2d2v/`, but it is not wired into the registry.

The result is that BGK2D2V should currently be treated as an in-progress branch of the code base rather than an operational target.
