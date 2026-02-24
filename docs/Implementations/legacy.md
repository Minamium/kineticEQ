---
title: Legacy
parent: Implementations
nav_order: 36
---

# Legacy

## 対象

- `src/kineticEQ/_legacy/`

## 内容

旧クラスベース実装が残っている:

- `BGK1Dsim.py`
- `BGK1DPlotUtil.py`
- `BGK2D2V_core.py`
- `BGK2D2VPlotUtil.py`
- `BGK2Dsim.py`

## 現状

- トップレベル `kineticEQ.__init__` は legacy シンボルを一部公開 (`LegacyBGK1D`, `BGK1DPlotMixin`)。
- 新 API (`Config` / `Engine`) の主経路は `api/ + core/` 側。
- legacy は互換目的で残置され、開発の主対象ではない。

## 注意

- legacy 側には未実装箇所や古い backend path 参照がある。
- 新規利用は `api` / `core` 経路を推奨。
