"""
Run the policy to evaluate or to collect attention masks.

Author: Ian Char
Date: March 8, 2023
"""
import argparse
import os

import hydra
import gym
from omegaconf import OmegaConf
import torch
import numpy as np

import rlkit.envs
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.envs.env_utils import get_eval_settings_for_env


###########################################################################
# %% Load in arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True)
parser.add_argument('--num_rollouts', type=int, default=1)
parser.add_argument('--steps_per_rollout', type=int, default=1000)
args = parser.parse_args()

###########################################################################
# %% Load in the environment, settings, and policy.
###########################################################################
# Load in the config file.
cfg = OmegaConf.load(os.path.join(args.run_dir, 'config.yaml'))
# Load in the environment.
env_name = cfg['env']['expl_env']
if 'obs_type' in cfg['rl']:
    env_name += f'-{cfg["rl"]["obs_type"]}-v0'
if 'fusion-real' in env_name or 'bnrot-real' in env_name:
    import fusion_control.envs
env = gym.make(env_name)
observation_dim = env.observation_space.low.shape[0]
action_dim = env.action_space.low.shape[0]
# Load in the policy.
policy = hydra.utils.instantiate(
    cfg['rl']['policy'],
    obs_dim=observation_dim,
    action_dim=action_dim,
    encoder={'obs_dim': observation_dim, 'act_dim': action_dim},
)
policy.load_state_dict(
    torch.load(os.path.join(args.run_dir, 'policy.pt'), 'cpu'))
if isinstance(policy, TanhGaussianPolicy):
    policy = MakeDeterministic(policy)
else:
    policy.deterministic = True
    policy.train(False)
path_collector = hydra.utils.instantiate(
    cfg['rl']['eval_path_collector'],
    env=env,
    policy=policy,
    env_name=env_name,
)

###########################################################################
# %% Collect paths.
###########################################################################
paths = path_collector.collect_new_paths(
    args.steps_per_rollout,
    args.num_rollouts * args.steps_per_rollout,
    False,
)
returns = [np.sum(pth['rewards']) for pth in paths]
print(f'Returns {np.mean(returns)} +- {np.std(returns) / np.sqrt(len(paths))}')
