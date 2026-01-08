# api/simulation.py
class Engine:
    def __init__(self, config):
        self.config = config
        pass

    def step(self) -> None:
        ...

    def run(self) -> "Result":
        print(f"run {self.config}")
        

def run(config) -> "Result":
    return Engine(config).run()
