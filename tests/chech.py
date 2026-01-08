# APIのインポート
from kineticEQ import Config, Engine, run

# Configの作成
config = Config("test")

# 簡易実行
run(config)

# Engineのインスタンス作成
engine_instance = Engine(config)

# Engineからのrunメソッド実行
engine_instance.run()
