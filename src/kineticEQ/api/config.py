# api/config.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Type, Mapping, Any

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

# 計算するモデル
class Model(str, Enum):
    BGK1D1V = "BGK1D1V"
    BGK2D2V = "BGK2D2V"

# スキーム
class Scheme(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    HOLO = "holo"
    HOLO_NN = "holo_nn"

# 計算カーネルのバックエンド
class Backend(str, Enum):
    TORCH = "torch"
    CUDA_KERNEL = "cuda_kernel"

# 型指定による計算精度
class DType(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

# ログレベル
class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

# tqdm設定
class UseTqdm(str, Enum):
    TRUE = "true"
    FALSE = "false"

@dataclass(frozen=True)
class Config:
    model: str | Model = Model.BGK1D1V
    scheme: str | Scheme = Scheme.EXPLICIT
    backend: str | Backend = Backend.TORCH
    device: str = "cuda"
    dtype: str | DType = DType.FLOAT64
    log_level: str | LogLevel = LogLevel.INFO
    model_cfg: Any | None = None
    use_tqdm: str | UseTqdm = UseTqdm.TRUE
    benchlog: dict[str, Any] | None = None

    
    # 正規化処理
    def __post_init__(self):
        object.__setattr__(self, "model", parse_enum(Model, self.model,
                                 aliases={"bgk1d": Model.BGK1D1V,
                                          "bgk1d1v": Model.BGK1D1V,
                                          "bgk2d2v": Model.BGK2D2V}))
        object.__setattr__(self, "scheme", parse_enum(Scheme, self.scheme,
                                 aliases={"exp": Scheme.EXPLICIT,
                                          "imp": Scheme.IMPLICIT,
                                          "hl": Scheme.HOLO,
                                          "hl_nn": Scheme.HOLO_NN}))
        object.__setattr__(self, "backend", parse_enum(Backend, self.backend,
                                 aliases={"pytorch": Backend.TORCH,
                                          "cuda_backend": Backend.CUDA_KERNEL}))
        object.__setattr__(self, "dtype", parse_enum(DType, self.dtype,
                                aliases={"fp32": DType.FLOAT32,
                                         "fp64": DType.FLOAT64}))

        object.__setattr__(self, "device", str(self.device).strip())

        object.__setattr__(self, "use_tqdm", parse_enum(UseTqdm, self.use_tqdm,
                                aliases={"true": UseTqdm.TRUE,
                                         "false": UseTqdm.FALSE}))

        object.__setattr__(self, "log_level", parse_enum(LogLevel, self.log_level,
                                aliases={"debug": LogLevel.DEBUG,
                                         "info": LogLevel.INFO,
                                         "warning": LogLevel.WARNING,
                                         "err": LogLevel.ERROR}))

    @property
    def model_name(self) -> str:
        return self.model.value

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
    def log_level_name(self) -> str:
        return self.log_level.value

    @property
    def use_tqdm_name(self) -> str:
        return self.use_tqdm.value

    @property
    def use_tqdm_bool(self) -> bool:
        return self.use_tqdm == UseTqdm.TRUE

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model.value,
            "scheme": self.scheme.value,
            "backend": self.backend.value,
            "device": self.device,
            "dtype": self.dtype.value,
            "log_level": self.log_level.value,
            "model_cfg": self.model_cfg,
            "use_tqdm": self.use_tqdm.value,
        }
