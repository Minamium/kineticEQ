---
title: APIリファレンス
nav_order: 4
parent: kineticEQ Docs
has_children: true
---

# APIリファレンス

## トップレベルAPI

| クラス/関数 | 説明 |
|------------|------|
| `Config` | シミュレーション設定 |
| `Engine` | シミュレーションエンジン |
| `run()` | ワンライナー実行関数 |

## パラメータモジュール

### BGK1D

| クラス | 説明 |
|--------|------|
| `BGK1D.ModelConfig` | モデル全体の設定 |
| `BGK1D.Grid1D1V` | 空間・速度グリッド設定 |
| `BGK1D.TimeConfig` | 時間積分設定 |
| `BGK1D.BGK1D1VParams` | 物理パラメータ |

### スキーム固有パラメータ

| モジュール | クラス | 説明 |
|-----------|--------|------|
| `BGK1D.explicit` | `Params` | 陽解法パラメータ |
| `BGK1D.implicit` | `Params` | 陰解法パラメータ (Picard反復) |
| `BGK1D.holo` | `Params` | HOLO法パラメータ |

## 解析モジュール

| モジュール | 関数 |
|-----------|------|
| `analysis.BGK1D` | `run_benchmark`, `run_convergence_test`, `run_scheme_comparison_test` |
| `analysis.BGK1D.plotting` | `plot_benchmark_results`, `plot_convergence_results`, `plot_cross_scheme_results` |
