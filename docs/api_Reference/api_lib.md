---
title: kineticEQ.api
parent: API Reference
nav_order: 1
---

# API リファレンス

## トップレベル API

`kineticEQ` パッケージから直接インポートできるシンボル:

```python
from kineticEQ import Config, Engine, run, Result, BGK1D, BGK2D2V
```

### Config

シミュレーション全体の設定を保持する frozen dataclass。

```python
Config(
    model: str = "BGK1D1V",
    scheme: str = "explicit",
    backend: str = "torch",
    device: str = "cuda",
    dtype: str = "float64",
    log_level: str = "info",
    model_cfg: Any | None = None,
    use_tqdm: str = "true",
)
```

詳細は [Engine Overview](../getting-started/Engine_overview.md) を参照。

### Engine

```python
Engine(config: Config, apply_logging_flag: bool = True)
```

- `config` -- `Config` インスタンス
- `apply_logging_flag` -- `True` で Python logging を設定に基づいて自動構成する

#### Engine.run()

```python
engine.run() -> Result
```

時間発展ループを実行し `Result` を返す。

### run

```python
run(config: Config) -> Result
```

`Engine(config).run()` と等価なワンライナー関数。

### Result

```python
@dataclass
class Result:
    metrics: dict[str, float] | None = None
    payload: dict[str, Any] | None = None
```

---

## パラメータモジュール -- BGK1D

`kineticEQ.params.BGK1D` から公開されるクラス群。
`from kineticEQ import BGK1D` でアクセスできる。

### BGK1D.ModelConfig

```python
BGK1D.ModelConfig(
    grid: Grid1D1V = Grid1D1V(),
    time: TimeConfig = TimeConfig(),
    params: BGK1D1VParams = BGK1D1VParams(),
    initial: InitialCondition1D = InitialCondition1D(),
    scheme_params: Any = None,
)
```

### BGK1D.Grid1D1V

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `nx` | int | 124 | 空間格子点数 |
| `nv` | int | 64 | 速度格子点数 |
| `Lx` | float | 1.0 | 空間領域長 |
| `v_max` | float | 10.0 | 速度空間上限 $[-v_{\max}, v_{\max}]$ |

### BGK1D.TimeConfig

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `dt` | float | 5e-4 | 時間刻み幅 |
| `T_total` | float | 0.05 | 総シミュレーション時間 |

プロパティ `n_steps` は `ceil(T_total / dt)` を返す。

### BGK1D.BGK1D1VParams

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `tau_tilde` | float | 0.5 | 無次元緩和時間 |

### BGK1D.InitialCondition1D

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `initial_regions` | tuple[dict, ...] | 2 領域 Sod 問題 | 区分的定数の初期モーメント。各要素は `x_range`, `n`, `u`, `T` を持つ辞書 |

### BGK1D.explicit.Params

追加パラメータなし（空の dataclass）。

### BGK1D.implicit.Params

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `picard_iter` | int | 16 | Picard 反復の最大回数 |
| `picard_tol` | float | 1e-4 | 相対収束判定閾値 |
| `abs_tol` | float | 1e-16 | 絶対収束判定閾値 |
| `moments_cnn_modelpath` | str \| None | None | CNN warmstart チェックポイントパス |

### BGK1D.holo.Params

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `ho_iter` | int | 8 | HO ループ最大反復数 |
| `ho_tol` | float | 1e-4 | HO 相対収束閾値 |
| `ho_abs_tol` | float | 1e-12 | HO 絶対収束閾値 |
| `lo_iter` | int | 16 | LO ループ最大反復数 |
| `lo_tol` | float | 1e-4 | LO 相対収束閾値 |
| `lo_abs_tol` | float | 1e-12 | LO 絶対収束閾値 |
| `Con_Terms_do` | bool | True | 整合項を有効にする |
| `flux_consistency_do` | bool | True | フラックス整合を有効にする |

---

## パラメータモジュール -- BGK2D2V

`from kineticEQ import BGK2D2V` でアクセスできる。

### BGK2D2V.Grid2D2V

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `nx` | int | 124 | x 方向空間格子点数 |
| `ny` | int | 124 | y 方向空間格子点数 |
| `nx_v` | int | 16 | $v_x$ 方向速度格子点数 |
| `ny_v` | int | 16 | $v_y$ 方向速度格子点数 |
| `Lx` | float | 1.0 | x 方向空間領域長 |
| `Ly` | float | 1.0 | y 方向空間領域長 |
| `v_max` | float | 10.0 | 速度空間上限 |

### BGK2D2V.TimeConfig

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `dt` | float | 5e-3 | 時間刻み幅 |
| `T_total` | float | 0.05 | 総シミュレーション時間 |

### BGK2D2V.BGK2D2VParams

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `tau_tilde` | float | 0.5 | 無次元緩和時間 |

---

## 解析モジュール

`kineticEQ.analysis.BGK1D` から公開される解析・ベンチマーク関数群。

### 実行関数

| 関数 | 説明 |
|------|------|
| `run_benchmark(...)` | 実行速度ベンチマーク |
| `run_convergence_test(...)` | 格子収束テスト |
| `run_scheme_comparison_test(...)` | スキーム間比較テスト |

### プロット関数

`kineticEQ.analysis.BGK1D.plotting` から公開:

| 関数 | 説明 |
|------|------|
| `plot_benchmark_results(...)` | ベンチマーク結果の可視化 |
| `plot_convergence_results(...)` | 収束テスト結果の可視化 |
| `plot_cross_scheme_results(...)` | スキーム比較結果の可視化 |
| `plot_timing_benchmark(...)` | タイミングベンチマーク可視化 |
