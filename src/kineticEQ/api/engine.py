# api/engine.py
from __future__ import annotations
from .result import Result
from .config import Config

class Engine:
    def __init__(self, config: Config):
        self.config = config

        print(f"config: {self.config.as_dict}")


    def step(self) -> None:
        ...

    def run(self) -> Result:

        # テストプリント
        print(f"run {self.config.scheme_name} {self.config.backend_name}")

        return Result()

def run(config) -> Result:
    return Engine(config).run()
