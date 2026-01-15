---
title: kineticEQ Docs
nav_order: 1
---

# What is kineticEQ?

kineticEQ is a kinetic equation solver using torch library.

## Quick Start

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128),
        time=BGK1D.TimeConfig(dt=1e-6, T_total=0.01),
    )
)

engine = Engine(cfg)
engine.run()
```

## Documentation

- [Installation](getting-started/installation.md)
- [Tutorial](tutorials/basic.md)
- [API Reference](api/index.md)
