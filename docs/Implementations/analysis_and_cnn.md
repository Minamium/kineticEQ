---
title: Analysis And CNN
parent: Implementations
nav_order: 35
---

# Analysis And CNN

## `analysis/BGK1D`

### 実行関数

- `benchmark.py` -> `run_benchmark`
- `convergence_test.py` -> `run_convergence_test`
- `scheme_comparison.py` -> `run_scheme_comparison_test`

### plotting

- `plot_benchmark_result.py`
- `plot_timing_benchmark.py`
- `plot_convergence_result.py`
- `plot_scheme_comparison_result.py`
- `plot_moment_cnn_test.py`

### utilities

- `utils/snapshot.py`
- `utils/compute_err.py`
- `utils/swap_grid_params.py`

## `CNN/BGK1D1V`

### 主なファイル

- `models.py`: `MomentCNN1D`, `LiteResBlock1D`
- `dataloader_npz.py`: `BGK1D1VNPZDeltaDataset`
- `losses.py`: 標準化 residual + shock mask loss
- `train.py`: 学習本体
- `multi_train.py`: sweep orchestration
- `eval_warmstart_debug.py`: warmstart 効果評価
- `generate_bgk1d_implicit_dataset.py`: データ生成
- `build_manifest.py`: NPZ manifest 生成

### 実装連携点

`core/schemes/BGK1D/bgk1d_implicit_cuda_kernel.py` から:

- checkpoint ロード (`load_moments_cnn_model`)
- 予測 (`predict_next_moments_delta`)
- Picard 初期モーメントへの注入

## `plotting/bgk1d`

- `plot_state.py`: 1ステップ前後の `state` 可視化
