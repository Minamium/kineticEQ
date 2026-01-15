---
title: インストール
nav_order: 2
parent: kineticEQ Docs
---

# インストール

## 必要条件

- Python >= 3.10
- PyTorch >= 2.0
- CUDA Toolkit (GPU使用時)

## pipでインストール

```bash
pip install kineticEQ
```

## 開発版のインストール

```bash
git clone https://github.com/Minamium/kineticEQ.git
cd kineticEQ
pip install -e .
```

## CUDAカーネルのビルド

CUDAカーネルは初回実行時に自動でJITコンパイルされる。
事前ビルドする場合:

```bash
python -c "from kineticEQ.cuda_kernel import compile; compile.compile_all()"
```
