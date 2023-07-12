import os
import pickle as pkl

from gym.spaces import Box, Discrete, Tuple

from rlkit import RLKIT_PROJECT_PATH

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


def mode(env, mode_type):
    try:
        getattr(env, mode_type)()
    except AttributeError:
        pass


def get_eval_settings_for_env(env_name, split='val'):
    base_path = os.path.join(RLKIT_PROJECT_PATH, 'rlkit/envs/eval_settings')
    file_name = None
    if split not in ('tr', 'val', 'te'):
        raise ValueError(f'Unrecognized split type {split}')
    if ('msd' not in env_name and 'nav' not in env_name
            and 'fusion' not in env_name and 'bnrot' not in env_name):
        raise ValueError(f'Unknown environment {env_name}')
    file_name = '-'.join(env_name.split('-')[:2] + [f'{split}.pkl'])
    with open(os.path.join(base_path, file_name), 'rb') as f:
        settings = pkl.load(f)
    return settings
