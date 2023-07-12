"""
Set up things for training.

Author: Ian Char
Date: Feb 16, 2023
"""
import gym
import hydra

import rlkit.envs
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy


def construct_sac(
    env_cfg,
    rl_cfg,
    save_every=0,
):
    # Create the necessary environments.
    # if sum(['fusion-real' in v for v in env_cfg['eval_envs'].values()]) > 0:
    #     import fusion_control.envs
    # if sum(['bnrot-real' in v for v in env_cfg['eval_envs'].values()]) > 0:
    #     import fusion_control.envs
    expl_env = gym.make(env_cfg['expl_env'])
    observation_dim = expl_env.observation_space.low.shape[0]
    action_dim = expl_env.action_space.low.shape[0]
    eval_envs = {k: gym.make(v) for k, v in env_cfg['eval_envs'].items()}
    # Set up logging.
    if save_every > 0:
        setup_logger(
            exp_prefix='rlkit',
            snapshot_mode='gap_and_last',
            snapshot_gap=save_every,
            log_dir='./',
        )
    else:
        setup_logger(exp_prefix='rlkit', log_dir='./')
    # Instantiate the networks.
    if rl_cfg.get('share_encoders', False):
        raise NotImplementedError('TODO')
    policy = hydra.utils.instantiate(
        rl_cfg['policy'],
        obs_dim=observation_dim,
        action_dim=action_dim,
        encoder={'obs_dim': observation_dim, 'act_dim': action_dim},
    )
    q_nets = []
    for qidx in range(4):
        q_nets.append(hydra.utils.instantiate(
            rl_cfg['qnet'],
            input_size=observation_dim + action_dim,
            output_size=1,
            obs_dim=observation_dim,
            act_dim=action_dim,
            encoder={'obs_dim': observation_dim, 'act_dim': action_dim},
        ))
    # Make the path collectors and replay buffer.
    eval_policy = (MakeDeterministic(policy) if isinstance(policy, TanhGaussianPolicy)
                   else policy)
    eval_path_collectors = {k: hydra.utils.instantiate(rl_cfg['eval_path_collector'],
                                                       env=eval_envs[k],
                                                       env_name=v,
                                                       policy=eval_policy)
                            for k, v in env_cfg['eval_envs'].items()}
    expl_path_collector = hydra.utils.instantiate(
        rl_cfg['expl_path_collector'],
        env=expl_env,
        policy=policy,
    )
    replay_buffer = hydra.utils.instantiate(rl_cfg['replay_buffer'], env=expl_env)
    # Instantiate the trainer and algorithm.
    trainer = hydra.utils.instantiate(
        rl_cfg['trainer'],
        env=expl_env,
        policy=policy,
        qf1=q_nets[0],
        qf2=q_nets[1],
        target_qf1=q_nets[2],
        target_qf2=q_nets[3],
    )
    algorithm = hydra.utils.instantiate(
        rl_cfg['algorithm'],
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_envs,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collectors,
        replay_buffer=replay_buffer,
    )
    return algorithm, policy, expl_env
