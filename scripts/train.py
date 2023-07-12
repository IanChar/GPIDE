"""
Train policy on environment.

Author: Ian Char
Date: February 16, 2023
"""
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from rlkit import RLKIT_PROJECT_PATH
import rlkit.torch.pytorch_util as ptu
from rlkit.util.training_setup import construct_sac


@hydra.main(config_path=f'{RLKIT_PROJECT_PATH}/cfgs', config_name='train')
def train(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, 'config.yaml')
    if 'seed' in cfg:
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
    if cfg.get('cuda_device', None) is not None:
        ptu.set_gpu_mode(True, cfg['cuda_device'])
    # If we are dealing with an msd environment take observation type into account.
    if ('msd' in cfg['env']['expl_env']
            or 'nav' in cfg['env']['expl_env']
            or 'fusion' in cfg['env']['expl_env']
            or 'bnrot' in cfg['env']['expl_env']):
        with open_dict(cfg):
            cfg['env']['expl_env'] = '-'.join([
                cfg['env']['expl_env'],
                cfg['rl']['obs_type'],
                'v0',
            ])
            for k, v in cfg['env']['eval_envs'].items():
                cfg['env']['eval_envs'][k] = '-'.join([
                    v,
                    cfg['rl']['obs_type'],
                    'v0',
                ])
    if cfg['rl']['rl_algorithm'] == 'sac':
        constructor = construct_sac
    algorithm, policy, env = constructor(cfg['env'], cfg['rl'],
                                         cfg.get('save_every', 0))
    algorithm.to(ptu.device)
    print(OmegaConf.to_yaml(cfg))
    if hasattr(policy, 'encoder'):
        encoder_params = sum([p.numel() for p in policy.encoder.parameters()
                              if p.requires_grad])
        print('=' * 50)
        print(f'Number Policy Encoder Params: {encoder_params}')
        print('=' * 50)
    if cfg.get('debug', False):
        breakpoint()
    if cfg.get('torch_anomaly', False):
        torch.autograd.set_detect_anomaly(True)
    algorithm.train()
    # Save off policy.
    os.makedirs('policy')
    torch.save(policy.state_dict(), 'policy/weights.pt')
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.low.shape[0]
    if 'policy' in cfg['rl']:
        with open_dict(cfg):
            cfg['rl']['policy']['obs_dim'] = obs_dim
            cfg['rl']['policy']['action_dim'] = act_dim
            if 'encoder' in cfg['rl']['policy']:
                cfg['rl']['policy']['encoder']['obs_dim'] = obs_dim
                cfg['rl']['policy']['encoder']['act_dim'] = act_dim
        OmegaConf.save(cfg['rl']['policy'], 'policy/config.yaml')


if __name__ == '__main__':
    train()
