# APIのインポート
from kineticEQ import Config, Engine, run

# Configの作成
config = Config()

# 簡易実行
run(config)

# コンフィグの編集
from dataclasses import replace
config = replace(config, scheme="implicit", backend="cuda_kernel", device="cuda", dtype="float64")


# Engineのインスタンス作成
engine_instance = Engine(config)

# Engineからのrunメソッド実行
engine_instance.run()
