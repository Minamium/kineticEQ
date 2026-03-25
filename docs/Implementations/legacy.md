---
title: Legacy
parent: Implementations
nav_order: 36
lang: ja
---

# Legacy

## 対象

- `src/kineticEQ/_legacy/`

## 内容

旧クラスベース実装が残されている。

- `BGK1Dsim.py`
- `BGK1DPlotUtil.py`
- `BGK2D2V_core.py`
- `BGK2D2VPlotUtil.py`
- `BGK2Dsim.py`

## 現状の位置づけ

- 新しい主経路は `api/ + core/ + params/` にある。
- legacy は互換性維持と参照用コードとして残置されている。
- 旧ノートブックや旧可視化ロジックを追う場合にのみ参照するのがよい。

## 注意

- backend 名称や責務分離が現行 API と一致しない箇所がある。
- 新規実装や性能評価は legacy ではなく現行 stepper 群を基準に行うべきである。
