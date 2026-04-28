---
title: Analysis And CNN
parent: Implementations
nav_order: 35
lang: ja
---

# Analysis And CNN

## `analysis/BGK1D1V`

### 実行モジュール

- `benchmark.py`: `run_benchmark`
- `convergence_test.py`: `run_convergence_test`
- `scheme_comparison.py`: `run_scheme_comparison_test`

### plotting

- `plot_benchmark_result.py`
- `plot_timing_benchmark.py`
- `plot_convergence_result.py`
- `plot_scheme_comparison_result.py`
- `plot_moment_cnn_test.py`
- `plot_moment_cnn_test_v2.py`

### utilities

- `utils/snapshot.py`
- `utils/compute_err.py`
- `utils/swap_grid_params.py`

## `CNN/BGK1D1V`

現行 CNN 実装は、単一スクリプト群ではなく、`gen_traindata_v1` / `gen_traindata_v2` / `train` / `evaluation` / `util` に分割されている。

### `util/`

- `models.py`: `MomentCNN1D`, `LiteResBlock1D`
- `input_state.py`: 入力状態の正規化、チャネル構成、`prev_delta` モード
- `losses.py`: 標準化残差 `r_n, r_u, r_T` と shock mask 損失

### `gen_traindata_v1/`

- NPZ ベースの旧データパイプライン
- `generate_bgk1d_implicit_dataset.py`: implicit rollout から `(n,u,T)` 時系列を保存
- `build_manifest.py`: case 単位 split を持つ manifest を生成
- `dataloader_npz.py`: `BGK1D1VNPZDeltaDataset`

### `gen_traindata_v2/`

- shard 化された PT ベースの新データパイプライン
- `generate_bgk1d_implicit_dataset.py`: PT shard を生成
- `manifest.py`: `dataset_manifest.json` と `case_manifest.jsonl` を組み立てる
- `dataloader_pt.py`: `BGK1D1VShardDeltaDataset`

### `train/`

- `train.py`: 学習本体
- `multi_train.py`: JSON 設定に基づく sweep 実行

`train.py` は `dnu` / `dw` target、`input_state_type`、`input_temporal_mode`、train-time warm-eval、AA 関連 override などを受け取り、checkpoint metadata に学習条件を保存する。
現在は `conv_type` も train 引数として公開されており、train-time warm-eval の implicit 設定へ転送される。

### `evaluation/`

- `engine.py`: warm-eval 実行の中核
- `spec.py`: case / target / cache policy の dataclass
- `train_eval.py`: 学習時評価の front-end
- `eval_warmstart_debug.py`: 互換ラッパ
- `legacy/eval_warmstart_debug.py`: 従来 CLI

### core との接続

`core/schemes/BGK1D1V/bgk1d_implicit_*` からは、以下の経路で CNN が利用される。

1. `load_moments_cnn_model()` で checkpoint をロード
2. `predict_next_moments_delta()` で内部セルの次時刻モーメント増分を予測
3. `ws.n`, `ws.nu`, `ws.T` の初期値を補正し、Picard 反復開始点として用いる

### 学習対象の意味

- `delta_type="dw"`: `(\Delta n, \Delta u, \Delta T)` を学習する legacy primitive increment
- `delta_type="dnu"`: `(\Delta n, \Delta \nu, \Delta T)` を学習する conservative increment

現行実装では `dnu` が物理保存量との整合を取りやすい標準設定である。
