TRON1 Reinforcement Learning
===========================

Overview
--------
This folder contains two training paths for the TRON1 balancing task:

- MuJoCo + Stable-Baselines3 PPO
- MJX + custom JAX PPO

The recent stable MJX baseline is the `basic` path, which uses a simple reward:

- height maintenance reward
- upright reward
- fall penalty


What to commit
--------------
This repository is set up to keep source code and experiment notes in Git, while
excluding large training artifacts such as:

- `*.pkl`
- `*.zip`
- checkpoint directories
- TensorBoard/log directories
- Python cache files

The large training artifacts are ignored.  The simulation asset folder
`tron1-mujoco-sim/` is also not committed here because it is an upstream
repository on its own and is fetched separately on the target machine.


Important files
---------------
- `train.py`: MuJoCo + SB3 PPO training
- `run.py`: MuJoCo policy viewer
- `tron1_env.py`: MuJoCo Gym environment
- `train_mjx.py`: MJX training with the experimental reward versions
- `run_mjx.py`: MJX viewer for the main MJX path
- `tron1_mjx_env.py`: MJX environment for the main path
- `train_mjx_basic.py`: MJX training with the simple balance reward
- `run_mjx_basic.py`: MJX viewer for the simple balance path
- `tron1_mjx_env_basic.py`: MJX environment for the simple balance path
- `MJX_EXPERIMENTS.txt`: version-by-version experiment notes
- `tron1-mujoco-sim/`: external dependency directory expected at runtime


Recommended handoff target
--------------------------
For continuing the current work on another machine, start from:

- `train_mjx_basic.py`
- `run_mjx_basic.py`

Current default basic run name:

- `basic_v2`


Environment setup
-----------------
Example with conda:

```bash
conda create -n mjx python=3.12 -y
conda activate mjx
pip install --upgrade pip
pip install mujoco gymnasium jax jaxlib numpy
```

If the desktop uses NVIDIA GPU acceleration with JAX, install the matching CUDA
build of JAX for that machine instead of plain CPU `jaxlib`.

If you also want to run the original MuJoCo + SB3 path:

```bash
pip install stable-baselines3 tensorboard
```

Simulation assets
-----------------
This project expects the upstream simulator repository at:

```bash
cd tron1
git clone https://github.com/limxdynamics/tron1-mujoco-sim.git
```

After cloning, the following path should exist:

```bash
tron1/tron1-mujoco-sim/robot-description/pointfoot/PF_TRON1A/xml/robot.xml
```


Training commands
-----------------
MJX basic training:

```bash
cd tron1
python train_mjx_basic.py
```

MJX main training:

```bash
cd tron1
python train_mjx.py
```

MuJoCo + SB3 training:

```bash
cd tron1
python train.py
```


Viewer commands
---------------
MJX basic final model:

```bash
cd tron1
python run_mjx_basic.py
```

MJX basic best model:

```bash
cd tron1
python run_mjx_basic.py tron1_balance_basic_v2_best.pkl
```

MJX main model:

```bash
cd tron1
python run_mjx.py
```


Notes
-----
- On the laptop setup used during development, GPU runs often used:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
```

- The `warp` import warning is not required for the current JAX/MJX training
  path and can usually be ignored.
- Model files are intentionally not committed.  Re-train on the desktop or move
  model artifacts separately if needed.
