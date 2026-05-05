---
title: Installation
parent: Getting Started
nav_order: 11
lang: ja
---

# Installation

## 実行要件

- Python 3.10 以上
- PyTorch 2.0 以上
- `cuda_kernel` を利用する場合は CUDA 対応 GPU、CUDA Toolkit 12.x 以上、`nvcc`
- CUDA 拡張の JIT コンパイルでは GCC/G++ 9 以上を推奨
- `cpu_kernel` を利用する場合は C++17 を扱えるホストコンパイラ

NVHPC は CUDA Toolkit 側で対応する host compiler として利用できる場合がある。ただし kineticEQ の CUDA 拡張は PyTorch C++ extension としてビルドされるため、推奨・検証対象は `nvcc` + GCC/G++ である。

## 依存パッケージ

`pyproject.toml` で要求される主要依存は以下である。

- `torch>=2.0`
- `numpy`
- `scipy`
- `tqdm`
- `ninja`
- `setuptools`
- `wheel`
- `zarr`
- `numcodecs`

可視化を併用する場合は、`viz` extra として `matplotlib` および `plotly>=5` を導入する。

## インストール

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e ".[viz]"
```

## 拡張モジュールのビルド

`kineticEQ` の高速 backend は、`torch.utils.cpp_extension.load` を用いた JIT コンパイルでロードされる。したがって、初回起動時には拡張ビルドが発生する。

### CUDA 拡張

`src/kineticEQ/cuda_kernel/compile.py` では、以下のローダが定義されている。

- `load_explicit_fused()`
- `load_implicit_fused()`
- `load_gtsv()`
- `load_lo_blocktridiag()`
- `load_implicit_AA()`

BGK1D の `cuda_kernel` 経路では、fused binding が `torch.float64` を要求するため、`dtype="float64"` を前提に設定する必要がある。

JIT コンパイル時の host compiler とビルドキャッシュは、必要に応じて以下の環境変数で固定できる。`TORCH_EXTENSIONS_DIR` と `MAX_JOBS` は利用環境に合わせて任意のキャッシュ先と並列コンパイル数に置き換える。

```bash
# Optional: adjust these values for your compiler and build-cache environment.
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export TORCH_EXTENSIONS_DIR=/path/to/your/torch_extensions
export MAX_JOBS=8
```

- `CC` / `CXX`: PyTorch C++ extension が利用する C/C++ compiler
- `CUDAHOSTCXX`: `nvcc` が利用する host C++ compiler
- `TORCH_EXTENSIONS_DIR`: 任意の JIT build cache 出力先
- `MAX_JOBS`: 任意の並列コンパイル数

### CPU 拡張

`src/kineticEQ/cpu_kernel/compile.py` では、implicit 用に以下が定義されている。

- `load_implicit_fused_cpu()`
- `load_gtsv_cpu()`

こちらも binding 側で `float64` を要求するため、`backend="cpu_kernel"` を用いる implicit 実行では `dtype="float64"` を指定する。

## 拡張キャッシュ

ビルドキャッシュの出力先は `TORCH_EXTENSIONS_DIR` で変更できる。

```bash
export TORCH_EXTENSIONS_DIR="/path/to/torch_extensions"
```

HPC 環境では、ジョブ投入前にログインノードで一度ロードしておくと、初回ビルドの待ち時間を分離しやすい。

```python
from kineticEQ.cuda_kernel.compile import (
    load_explicit_fused,
    load_implicit_fused,
    load_gtsv,
    load_lo_blocktridiag,
    load_implicit_AA,
)
from kineticEQ.cpu_kernel.compile import (
    load_implicit_fused_cpu,
    load_gtsv_cpu,
)
```

## 運用上の注意

- `device="cuda"` は `torch.cuda.is_available()` を通過した場合にのみ許容される。
- `device="mps"` も `resolve_device()` により検証される。
- `cuda_kernel` と `cpu_kernel` は PyTorch テンソルを直接受け取るため、dtype・device・contiguous 条件が満たされないと binding で例外になる。
