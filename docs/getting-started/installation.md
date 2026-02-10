---
title: Installation
parent: Getting Started
nav_order: 11
---

# Installation

## 動作要件

- **Python** >= 3.10
- **CUDA 対応 GPU** (`cuda_kernel` バックエンドを使う場合)
- **C++ コンパイラ** (gcc / g++) -- CUDA カーネルの JIT コンパイルに必要

## 依存パッケージ

`pyproject.toml` で宣言されている必須依存:

| パッケージ | 用途 |
|-----------|------|
| `torch` >= 2.0 | テンソル演算、CUDA バックエンド |
| `numpy` | 配列操作 |
| `scipy` | 数値計算ユーティリティ |
| `tqdm` | 進捗バー表示 |
| `ninja` | CUDA カーネル JIT コンパイルの高速化 |
| `setuptools`, `wheel` | ビルド |
| `zarr`, `numcodecs` | データセット I/O |

オプション依存 (`pip install kineticEQ[viz]`):

| パッケージ | 用途 |
|-----------|------|
| `matplotlib` | プロット |
| `plotly` >= 5 | インタラクティブ可視化 |

## インストール

### 開発インストール（推奨）

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e ".[viz]"
```

### CUDA カーネルについて

`cuda_kernel` バックエンドを使用すると、初回実行時に `torch.utils.cpp_extension` を通じて CUDA カーネルが JIT コンパイルされる。
コンパイル済みキャッシュはデフォルトで `~/.cache/torch_extensions/` に保存される。

環境変数 `TORCH_EXTENSIONS_DIR` でキャッシュの保存先を変更できる:

```bash
export TORCH_EXTENSIONS_DIR="/path/to/cache"
```

### HPC クラスタでの注意点

- 計算ノードからインターネットに接続できない場合、ログインノードで事前に `pip install` を完了しておく
- JIT コンパイルキャッシュをジョブ投入前に生成しておくと、walltime の無駄を防げる:

```python
from kineticEQ.cuda_kernel.compile import load_all_kernels
```

- `TORCH_CUDA_ARCH_LIST` 環境変数で対象 GPU アーキテクチャを指定する（例: V100 は `"7.0"`, A100 は `"8.0"`）