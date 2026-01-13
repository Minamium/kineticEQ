# kineticEQ/utillib/pretty.py
from __future__ import annotations
from dataclasses import is_dataclass, asdict
from typing import Any, Iterable

def _to_mapping(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x

def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        v = _to_mapping(v)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def _sort_keys(keys: list[str], key_order: Iterable[str] | None) -> list[str]:
    if key_order is None:
        return sorted(keys)

    order = list(key_order)

    def matches(p: str, k: str) -> bool:
        # p が "xxx." で終わるときだけ prefix 扱い、それ以外は完全一致
        if p.endswith("."):
            return k.startswith(p)
        return k == p

    def rank(k: str) -> tuple[int, int, str]:
        best_i = len(order)
        best_plen = -1
        for i, p in enumerate(order):
            if matches(p, k):
                plen = len(p)
                if i < best_i or (i == best_i and plen > best_plen):
                    best_i = i
                    best_plen = plen
        return (best_i, -best_plen, k)

    return sorted(keys, key=rank)

def format_kv_block(
    obj: Any,
    indent: int = 2,
    key_order: Iterable[str] | None = None,
    exclude: set[str] | None = None,
) -> str:
    """
    dataclass/dict をフラット化して "key: value" のブロック文字列を返す。
    - タイトル行は出さない（呼び出し側で付ける）
    - key_order が None のとき、obj.__pretty_order__ があればそれを使う
    """
    if key_order is None:
        key_order = getattr(obj, "__pretty_order__", None)

    base = dict(_to_mapping(obj))
    if not isinstance(base, dict):
        return repr(obj)
    
    if exclude:
        for k in exclude:
            base.pop(k, None)

    flat = _flatten(base)
    pad = " " * indent
    lines = []
    for k in _sort_keys(list(flat.keys()), key_order):
        lines.append(f"{pad}{k:16s}: {flat[k]}")
    return "\n".join(lines)
