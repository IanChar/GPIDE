# GPIDE

This is code for the paper "PID-Inspired Inductive Biases for Deep Reinforcement
Learning in Partially Observable Control Tasks".

## Installation

This codebase was developed using python 3.9.16, and it is built off of
[rlkit](https://github.com/rail-berkeley/rlkit). In order to set up the code
run...

```
pip install -e .
```

## Training

In order to train run:

```
python scripts/train.py\
    name=test_run \
    rl=tracking_gpide_sac \
    env=msd-small \
    seed=0
```

Optionally, you can add `cuda_device=<cuda_device_number>` if you want to train
with a GPU. All options for `rl` and `env` can be found in `cfgs/rl` and `cfgs/env`.
The rl config specifies which RL baselines to use, and the env config specifies which
task to perform. When selecting a tracking environment, use an rl configuration
with the prefix "tracking" and with the prefix "pyb" otherwise.

## Making Tables and Plots

To make plots use `scripts/plotting/tracking_grid.py`
and `scripts/plotting/pyb_grid.py` for tracking and pybullet experiments, respectively.
Likewise, the `scripts/tables/tracking_table.py` and
`scripts/tables/pyb_table.py` can be used to generate tables.

In order to compare to all of the pybullet baselines, you must download the traces
provided in [pomdp-baselines](https://github.com/twni2016/pomdp-baselines). Extract
the .csv files from the pomdp environments and put them in a directory named `pomdp`
at the top level of this directory.

## Traces from Paper

All performance traces can be downloaded [here](https://drive.google.com/file/d/1V-kgqpcSZolWVBPGHZmPbIn0ZZqTRNdr/view?usp=sharing).
