# api/config.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Type, Mapping

E = TypeVar("E", bound=Enum)

def parse_enum(
    enum_cls: Type[E],
    x: str | E,
    *,
    aliases: Mapping[str, E] | None = None) -> E:
    # Enumならそのまま
    if isinstance(x, enum_cls):
        return x

    s = str(x).strip().lower()

    # alias
    if aliases is not None and s in aliases:
        return aliases[s]

    # Enum.value一致
    for e in enum_cls:
        if s == str(e.value).lower():
            return e

    allowed = [str(e.value) for e in enum_cls]
    if aliases:
        allowed += [f"{k}->{v.value}" for k, v in aliases.items()]
    raise ValueError(f"unknown {enum_cls.__name__}: {x!r}. allowed: {allowed}")

class Scheme(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    HOLO = "holo"

class Backend(str, Enum):
    TORCH = "torch"
    CUDA_KERNEL = "cuda_kernel"

class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

@dataclass(frozen=True)
class Config:
    scheme: str | Scheme = Scheme.EXPLICIT
    backend: str | Backend = Backend.TORCH
    device: str = "cuda"
    dtype: str | DType = DType.FLOAT64
    
    # 正規化処理
    def __post_init__(self):
        object.__setattr__(self, "scheme", parse_enum(Scheme, self.scheme,
                                 aliases={"exp": Scheme.EXPLICIT,
                                          "imp": Scheme.IMPLICIT,
                                          "hl": Scheme.HOLO}))
        object.__setattr__(self, "backend", parse_enum(Backend, self.backend,
                                 aliases={"pytorch": Backend.TORCH,
                                          "cuda_backend": Backend.CUDA_KERNEL}))
        object.__setattr__(self, "dtype", parse_enum(DType, self.dtype,
                                aliases={"fp32": DType.FLOAT32,
                                         "fp64": DType.FLOAT64}))

        object.__setattr__(self, "device", str(self.device).strip())

    @property
    def scheme_name(self) -> str:
        return self.scheme.value

    @property
    def backend_name(self) -> str:
        return self.backend.value

    @property
    def dtype_name(self) -> str:
        return self.dtype.value

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme.value,
            "backend": self.backend.value,
            "device": self.device,
            "dtype": self.dtype.value,
        }
