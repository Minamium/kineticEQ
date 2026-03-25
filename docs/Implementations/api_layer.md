---
title: API Layer
parent: Implementations
nav_order: 32
lang: ja
---

# API Layer

## 対象ディレクトリ

- `src/kineticEQ/api/`

## ファイルと責務

### `config.py`

- `Model`, `Scheme`, `Backend`, `DType`, `LogLevel`, `UseTqdm` の Enum 定義
- alias を含む `parse_enum()`
- `Config` dataclass と正規化ロジック
- `as_dict` や `*_name` 系アクセサ

### `engine.py`

- `default_model_cfg()` と `expected_model_cfg_type()` による `model_cfg` 補完・型検証
- `default_scheme_params()` による `scheme_params` 補完
- `apply_logging()` と `resolve_device()` の呼び出し
- `build_state()` / `build_stepper()` の統合
- progress bar を伴う時間発展ループ

### `result.py`

- `Result(metrics, payload)` dataclass

### `logging_utils.py`

- `kineticEQ` logger の handler / level 設定
- 既存 handler を再設定し、ログレベルだけが食い違う状態を避ける

## 実装上の要点

- `Engine` は `model_cfg.scheme_params` を参照するため、モデル dataclass 側にこのフィールドが無いと失敗する。
- 初期条件の設定は API 層ではなく stepper builder 層で行われる。
- `run(config)` は薄いラッパであり、API の本体はあくまで `Engine` である。
