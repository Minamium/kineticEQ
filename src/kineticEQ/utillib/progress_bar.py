# kineticEQ/utillib/progress_bar.py
"""progress_bar.py – toggle tqdm / text‑based progress output
===========================================================
BGK1Dsim など既存コードが使う最小 API を実装した軽量ヘルパ。

* **get_progress_bar(...)**  – `use_tqdm` が True なら `tqdm.tqdm`、
  False なら *SilentPB* を返す。
* **progress_write(msg, use_tqdm)** – `tqdm.write` か `print` を安全に呼び分け。
* **SilentPB** – 標準出力ベースの簡易バー。デフォルトでは *10 % 刻み* で
  「desc: xx % (elapsed 0.00 s)」を表示する。

この 3 つだけで BGK1Dsim 側は import 行を変えるだけで済む。
"""
from __future__ import annotations

import sys
import time
from types import ModuleType
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 0. 基本ユーティリティ
# ---------------------------------------------------------------------------

def _plain_write(msg: str) -> None:
    """標準出力への書き込み（flush 付き）。"""
    print(msg, file=sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# 1. SilentPB – tqdm が無い／使わないときの簡易バー
# ---------------------------------------------------------------------------
class SilentPB:
    """非常に軽量なダミー progress‑bar。

    * `update(n)` で進捗を受け取る。
    * **総ステップの 10 % 増えるごとに** 経過秒とともに行を出力する。
    * `with` 構文をサポート（何も返さずエラーにしない）。
    * `write()` は単純に `print()` ラッパ。
    """

    __slots__ = ("total", "n", "desc", "_start", "_next_tick", "_done")

    def __init__(self, total: Optional[int] = None, desc: str = "") -> None:
        self.total: Optional[int] = total if (total and total > 0) else None
        self.n: int = 0
        self.desc: str = desc
        self._start: float = time.perf_counter()
        # 10 % 刻みで表示。total が None の場合は表示しない。
        self._next_tick: float = 0.1 if self.total is not None else float("inf")
        self._done = False

    # --- context‑manager ----------------------------------------------------
    def __enter__(self) -> "SilentPB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 最終表示（100 %）がまだなら出す
        if self.total and self.n >= self.total and not self._done:
            self._emit(force=True)
            self._done = True

    # --- public API ---------------------------------------------------------
    def update(self, n: int = 1):
        self.n += n
        if self.total is None:
            return  # 総数不明 → 表示しない
        # 進捗率がしきい値を超えたら表示
        ratio = self.n / self.total
        if ratio >= self._next_tick or self.n >= self.total:
            self._emit()
            if self.n >= self.total:
                self._done = True

            # 次の 10 % しきい値へ
            self._next_tick += 0.1

            
    def close(self):
        pass  # 互換性のために存在

    def write(self, msg: str):
        _plain_write(msg)

    # --- internal -----------------------------------------------------------
    def _emit(self, *, force: bool = False):
        """進捗メッセージを出力。force=True なら割合にかかわらず表示。"""
        if self.total is None:
            return
        ratio = min(self.n / self.total, 1.0)
        if not force and ratio < self._next_tick - 0.0999:  # 少し余裕を見る
            return
        elapsed = time.perf_counter() - self._start
        percent = int(ratio * 100)
        msg = f"{self.desc}: {percent:3d}% (elapsed {elapsed:.2f} s)"
        _plain_write(msg)


# ---------------------------------------------------------------------------
# 2. tqdm をロードできない場合はダミーを作って NameError を防ぐ
# ---------------------------------------------------------------------------
try:
    import tqdm as _tqdm  # noqa: F401 – try real tqdm
    _have_tqdm = True
except ModuleNotFoundError:
    _have_tqdm = False
    dummy = ModuleType("tqdm")
    dummy.write = _plain_write  # type: ignore[attr-defined]
    sys.modules["tqdm"] = dummy

import tqdm  # type: ignore  # noqa: E402, F401  マジック：ここで必ず import 可能


# ---------------------------------------------------------------------------
# 3. 公開 API
# ---------------------------------------------------------------------------

def get_progress_bar(
    use_tqdm: bool,
    total: int,
    desc: str = "",
    **tqdm_kwargs: Any,
):
    """フラグ 1 本で real tqdm / SilentPB を返すヘルパ"""
    if use_tqdm and _have_tqdm:
        tqdm_kwargs.setdefault(
            "bar_format",
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        return tqdm.tqdm(total=total, desc=desc, **tqdm_kwargs)  # type: ignore[attr-defined]
    return SilentPB(total, desc)


def progress_write(msg: str, use_tqdm: bool = True) -> None:
    """tqdm.write or print を呼ぶ安全ラッパ"""
    if use_tqdm and _have_tqdm and hasattr(tqdm, "write"):
        tqdm.write(msg)  # type: ignore[attr-defined]
    else:
        _plain_write(msg)
