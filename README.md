# kineticEQ
Hello, Github.
This is a kinetic equation solver useing torch library.

## Installation

```bash
git clone https://github.com/Minamium/kineticEQ.git
pip install -e ./kineticEQ
```

## HPC Environment

### Slurm

```bash
pip install -e ./kineticEQ --no-deps --user
```


## Usage

```python
from kineticEQ import Config, Engine, run

Config_instance = Config()

Engine_instance = Engine(Config_instance)

Engine_instance.run()
