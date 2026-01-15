---
title: Installation
description: kineticEQ Docs - Installation
nav_order: 2
---

# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA Toolkit (GPU使用時)

## pip install

```bash
pip install kineticEQ
```

## Development install

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e .
```

## CUDA kernel build

CUDA kernel is automatically JIT compiled at the first run.

```bash
python -c "from kineticEQ.cuda_kernel import compile; compile.compile_all()"
```
