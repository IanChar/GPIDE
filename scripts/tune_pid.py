"""
Tune a PID controller on an environment.

Author: Ian Char
Date: April 19, 2023
"""
from collections import defaultdict
from functools import partial
import random

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from rlkit import RLKIT_PROJECT_PATH
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.path_collector import BatchTaskEvalCollector
from rlkit.policies.pid import PIDController, MultiPIDController
from rlkit.envs.env_utils import get_eval_settings_for_env


HORIZON = 100


def eval_pid_controller(
    controller,
    env,
    env_name,
    settings,
):
    path_collector = BatchTaskEvalCollector(
        env=env,
        env_name=env_name,
        policy=controller,
        split='N/A',
        settings=settings,
    )
    paths = path_collector.collect_new_paths(
        max_path_length=HORIZON,
        num_steps=100 ** 2,
        discard_incomplete_paths=True,
    )
    return np.mean(np.array([np.sum(path['rewards']) for path in paths]))


def oned_pid_opt_function(
    p: float,
    i: float,
    d: float,
    env: gym.Env,
    env_name: str,
    settings,
):
    controller = PIDController(p, i, d)
    return eval_pid_controller(controller, env=env, env_name=env_name,
                               settings=settings)


def twod_pid_opt_function(
    p1: float,
    p2: float,
    i1: float,
    i2: float,
    d1: float,
    d2: float,
    env: gym.Env,
    env_name: str,
    settings,
):
    controller = MultiPIDController([p1, p2],
                                    [i1, i2],
                                    [d1, d2])
    return eval_pid_controller(controller, env=env, env_name=env_name,
                               settings=settings)


@hydra.main(config_path=f'{RLKIT_PROJECT_PATH}/cfgs', config_name='tune_pid')
def tune_pid(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, 'config.yaml')
    if 'seed' in cfg:
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
    if cfg.get('cuda_device', None) is not None:
        ptu.set_gpu_mode(True, cfg['cuda_device'])
    # If we are dealing with a tracking environment take observation type into account.
    if 'fusion' in cfg['env']['expl_env'] or 'bnrot' in cfg['env']['expl_env']:
        import fusion_control.envs
    with open_dict(cfg):
        cfg['env']['expl_env'] = '-'.join([
            cfg['env']['expl_env'],
            'pid',
            'v0',
        ])
        for k, v in cfg['env']['eval_envs'].items():
            cfg['env']['eval_envs'][k] = '-'.join([
                v,
                'pid',
                'v0',
            ])
    # Create environment and load settings.
    env = gym.make(cfg['env']['expl_env'])
    tr_settings = get_eval_settings_for_env(cfg['env']['expl_env'], split='tr')
    # Get optimization ready.
    if ('dmsd' in cfg['env']['expl_env']
            or 'nav' in cfg['env']['expl_env']
            or 'bnrot' in cfg['env']['expl_env']):
        base_obj = twod_pid_opt_function
    else:
        base_obj = oned_pid_opt_function
    obj_function = partial(
        base_obj,
        env=env,
        env_name=cfg['env']['expl_env'],
        settings=tr_settings,
    )
    seed_offset = 0
    if 'small' in cfg['env']['expl_env'] or 'real' in cfg['env']['expl_env']:
        seed_offset = 100
    elif 'large' in cfg['env']['expl_env']:
        seed_offset = 200
    eval_dict = defaultdict(list)
    for seed in range(cfg['seed'] + seed_offset,
                      seed_offset + cfg['seed'] + cfg['num_seeds']):
        optimizer = BayesianOptimization(
            f=obj_function,
            pbounds=cfg['pid']['pbounds'],
            random_state=seed,
            allow_duplicate_points=True,
        )
        logger = JSONLogger(path="./tune_log.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(
            init_points=10,
            n_iter=cfg['pid']['capital'],
        )
        print(optimizer.max)
        # After finding the best, evaluate on all of the environments.
        eval_dict['training'].append(optimizer.max['target'])
        for etype, eval_env_name in cfg['env']['eval_envs'].items():
            te_settings = get_eval_settings_for_env(eval_env_name, split='te')
            eval_func = partial(
                base_obj,
                env=gym.make(eval_env_name),
                env_name=eval_env_name,
                settings=te_settings)
            eval_dict[etype].append(eval_func(**optimizer.max['params']))
    result_dict = {k: (np.mean(v), np.std(v) / np.sqrt(cfg['num_seeds']))
                   for k, v in eval_dict.items()}
    print(result_dict)
    with open('results.txt', 'w') as fl:
        for k, v in result_dict.items():
            fl.write(f'{k}: {v}\n')


if __name__ == '__main__':
    tune_pid()
