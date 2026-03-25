---
title: English Getting Started
nav_title: Getting Started
parent: English Docs
nav_order: 10
has_children: true
lang: en
---

# Getting Started

This section introduces the standard execution path, `Config -> Engine -> run()`. In the current implementation, initial conditions are not injected by `Engine` itself; they are applied inside each stepper builder, which is an important detail when tracing the runtime logic.

- [Installation](installation.md): requirements, dependencies, and JIT compilation of CUDA/CPU extensions
- [Engine Overview](Engine_overview.md): config normalization, model completion, and stepper construction
- [Examples](examples.md): representative explicit, implicit, holo, and CNN warm-start runs
