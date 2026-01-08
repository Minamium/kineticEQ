# api/result.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Result:
    metrics: Dict[str, float] | None = None
    payload: Dict[str, Any] | None = None
