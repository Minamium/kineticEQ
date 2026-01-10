# APIのインポート
from kineticEQ import Config, Engine, run

# Configの作成
config = Config(device="cpu",
                model="BGK1D1V")

# 簡易実行
print("========= execute run(config) Test =========")
run(config)
print("========= Test complete =========\n")

# コンフィグの編集
from dataclasses import replace
config = replace(config, 
                 model="BGK2D2V",
                 scheme="holo_nn", 
                 backend="cuda_kernel", 
                 device="cuda", 
                 dtype="float64", 
                 use_tqdm="false",
                 log_level="debug")


# Engineのインスタンス作成
print("========= Instance construction Test: Engine(config) =========")
engine_instance = Engine(config, apply_logging_flag=True)
print("========= Test complete =========\n")

# Engineからのrunメソッド実行
print("========= execute engine_instance.run() Test =========")
engine_instance.run()
print("========= Test complete =========\n")
