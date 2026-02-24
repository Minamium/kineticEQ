---
title: Installation
parent: Getting Started
nav_order: 11
---

# Installation

## 動作要件

- **Python** >= 3.10
- **PyTorch** >= 2.0
- `cuda_kernel` を使う場合:
  - CUDA 対応 GPU
  - nvcc / C++ コンパイラ

## 必須依存

`pyproject.toml`:

- `torch>=2.0`
- `numpy`
- `scipy`
- `tqdm`
- `ninja`
- `setuptools`
- `wheel`
- `zarr`
- `numcodecs`

オプション (`pip install -e ".[viz]"`):

- `matplotlib`
- `plotly>=5`

## インストール

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e ".[viz]"
```

## CUDA 拡張の JIT コンパイル

`backend="cuda_kernel"` の初回実行時に `torch.utils.cpp_extension.load` で拡張がビルドされる。

ビルド対象（`src/kineticEQ/cuda_kernel/compile.py`）:

- `load_explicit_fused()`
- `load_implicit_fused()`
- `load_gtsv()`
- `load_lo_blocktridiag()`
- `load_implicit_AA()`

`load_all_kernels()` は現行実装には存在しない。

キャッシュ先の変更:

```bash
export TORCH_EXTENSIONS_DIR="/path/to/cache"
```

## HPC での注意

- 事前にログインノードで依存を導入
- ジョブ前に必要カーネルを明示ロードしてキャッシュしておく

```python
from kineticEQ.cuda_kernel.compile import (
    load_explicit_fused,
    load_implicit_fused,
    load_gtsv,
    load_lo_blocktridiag,
    load_implicit_AA,
)
```

- `TORCH_CUDA_ARCH_LIST` を対象 GPU に合わせる
