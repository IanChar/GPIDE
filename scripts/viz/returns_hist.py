"""
Visualize rollouts of the MSD environment.
"""
import argparse
import os

import hydra
import gym
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt

import rlkit.envs
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.envs.env_utils import get_eval_settings_for_env
from rlkit.policies.pid import PIDController, MultiPIDController


###########################################################################
# %% Load in arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str)
parser.add_argument('--env', type=str, default='msd-fixed')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--pid_coefs', type=str, default=None)
args = parser.parse_args()

###########################################################################
# %% Load in the environment, settings, and policy.
###########################################################################
# Load in the config file.
if args.pid_coefs is None:
    cfg = OmegaConf.load(os.path.join(args.run_dir, 'config.yaml'))
    env_name = f'{args.env}-{cfg["rl"]["obs_type"]}-v0'
else:
    cfg = None
    env_name = f'{args.env}-pid-v0'
if 'fusion-real' in env_name or 'bnrot-real' in env_name:
    import fusion_control.envs
env = gym.make(env_name)
observation_dim = env.observation_space.low.shape[0]
action_dim = env.action_space.low.shape[0]
# Load in the policy.
if args.pid_coefs is None:
    policy = hydra.utils.instantiate(
        cfg['rl']['policy'],
        obs_dim=observation_dim,
        action_dim=action_dim,
        encoder={'obs_dim': observation_dim, 'act_dim': action_dim},
    )
    policy.load_state_dict(
        torch.load(os.path.join(args.run_dir, 'policy/weights.pt'), 'cpu'))
    if isinstance(policy, TanhGaussianPolicy):
        policy = MakeDeterministic(policy)
    else:
        policy.deterministic = True
        policy.train(False)
else:
    coefs = [float(gain) for gain in args.pid_coefs.split(',')]
    if len(coefs) == 3:
        policy = PIDController(*coefs)
    else:
        policy = MultiPIDController(
            p_gains=coefs[:2],
            i_gains=coefs[2:4],
            d_gains=coefs[4:],
        )
# Get the settings.
settings = get_eval_settings_for_env(env_name, split=args.split)
settings = {k: v for k, v in settings.items()}
rollouts = env.rollout(policy, len(settings['start']),
                       horizon=env.max_episode_steps, **settings)
returns = np.sum(rollouts['rewards'], axis=1)
plt.hist(returns, alpha=0.4, bins=25)
plt.axvline(np.mean(returns), ls=':', color='black')
plt.title(f'Score: {np.mean(returns):0.2f} '
          f'+- {np.std(returns) / np.sqrt(len(returns)):0.2f}')
plt.show()
