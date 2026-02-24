---
title: API Layer
parent: Implementations
nav_order: 32
---

# API Layer

## 対象ディレクトリ

- `src/kineticEQ/api/`

## ファイルと役割

- `config.py`
  - Enum 正規化 (`parse_enum`)
  - alias (`exp`, `imp`, `hl`, `hl_nn` など)
  - `Config` dataclass
- `engine.py`
  - `model_cfg` 補完
  - `scheme_params` 補完
  - device 検証
  - `build_state` / `build_stepper`
  - 時間ループ
- `result.py`
  - `Result(metrics, payload)`
- `logging_utils.py`
  - `kineticEQ` logger の level/handler 設定

## 実装上の注意

- `Engine.__init__` は `config.model_cfg.scheme_params` を参照するため、モデル側 `ModelConfig` に `scheme_params` が無いと失敗する。
- `run(config)` は `Engine(config).run()` の薄いラッパ。
