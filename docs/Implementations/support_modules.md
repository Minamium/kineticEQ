---
title: Support Modules
parent: Implementations
grand_parent: Japanese
nav_order: 37
lang: ja
---

# Support Modules

## `params/`

責務:

- BGK1D / BGK2D2V のモデル dataclass
- scheme 別既定 `scheme_params` の供給
- `expected_model_cfg_type()` による API 層との型整合

注意:

- `BGK2D2V.ModelConfig` は現状 `scheme_params` を持たないため、`Engine` 初期化と噛み合わない。

## `utillib/`

- `device_util.py`: `cuda` / `mps` / `cpu` の検証
- `progress_bar.py`: tqdm の有無を吸収する wrapper
- `pretty.py`: dataclass / dict をログ向け key-value 形式へ整形

## `plotting/`

- `plotting/bgk1d/plot_state.py`
  - `State1D1V` から `f`, `n`, `u`, `T` を可視化

## `tests/`

- `smoke/`: 最低限の実行確認
- `plot_test/`: plotting 呼び出し確認
- `bench_test/`: analysis API の呼び出し確認

補足:

- skip 条件を多く含むため、テストの不実行は直ちに不具合を意味しない。
- CUDA 非搭載環境では backend 依存テストの多くが skip される。
