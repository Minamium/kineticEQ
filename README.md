# kineticEQ
Hello, Github.

This is a kinetic equation solver useing torch library.

## Installation

```bash
git clone https://github.com/Minamium/kineticEQ.git
pip install -e ./kineticEQ
```

## Usage

```python
from kineticEQ.BGK1Dsim import BGK1D

sim = BGK1D({
        # chose solver
        #"solver": "implicit",
        "solver": "explicit",

        # chose implicit solver
        "implicit_solver": "cuSOLVER",
        #"implicit_solver": "tdma",
        #"implicit_solver": "full",

        # chose picard iteration
        "picard_iter": 1024,
        "picard_tol": 1e-6,

        # chose hyperparameter
        "tau_tilde": 5e-6,

        # chose numercial parameter
        "nx": 10000,
        "nv": 1000,
        "v_max": 10.0,
        "dt": 5e-6,

        "initial_regions": [
        {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8}
    ],

        # chose fixed moment boundary condition
        "n_left": 1.0,
        "u_left": 0.0,
        "T_left": 1.0,
        "n_right": 0.125,
        "u_right": 0.0,
        "T_right": 0.8,

        # chose simulation time
        "T_total":0.065,

        # chose device
        "device": "cuda",

        # chose dtype
        "dtype": "float64",

        # chose tqdm
        "use_tqdm":True
    }

sim.run_simulation()
'''