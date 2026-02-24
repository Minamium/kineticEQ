---
title: Support Modules
parent: Implementations
nav_order: 37
---

# Support Modules

## `params/`

責務:

- モデル設定 dataclass (`BGK1D`, `BGK2D2V`)
- `(model, scheme)` ごとの既定 `scheme_params` 供給 (`params/registry.py`)

注意:

- `BGK2D2V.ModelConfig` は現行 `scheme_params` を持たない

## `utillib/`

- `device_util.py`: `cuda` / `mps` / `cpu` の利用可否判定
- `progress_bar.py`: tqdm 有無を吸収する progress API
- `pretty.py`: dataclass/dict をログ向け key-value block へ整形

## `plotting/`

- `plotting/bgk1d/plot_state.py`
  - `State1D1V` から `f`, `n`, `u`, `T` を可視化

## `tests/`

- `smoke/`: 最低限の実行確認
- `plot_test/`: plotting 呼び出し確認
- `bench_test/`: analysis API の呼び出し確認

補足:

- test は構成・環境依存で skip 条件を多く含む（CUDA 非搭載環境など）。
