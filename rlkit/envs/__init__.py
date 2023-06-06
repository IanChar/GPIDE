import gym
from gym.envs.registration import register
import pybullet_envs

from rlkit.envs.msd_env import MassSpringDamperEnv
from rlkit.envs.double_msd_env import DoubleMassSpringDamperEnv
from rlkit.envs.fusion_toy_env import ToyFusionEnv
from rlkit.envs.betanrot_env import BetanRotEnv
from rlkit.envs.nav import Navigation
from rlkit.envs.unav import UNavigation
from rlkit.envs.wrappers import POMDPWrapper

###########################################################################
# Register all of the mass spring environments.
###########################################################################
obs_dict = {
    'xt': 'xt',
    'xtf': 'xtf',
    'xdtf': 'xdtf',
    'xpdtf': 'xpdtf',
    'xpitf': 'xpitf',
    'xidtf': 'xidtf',
    'pid': 'pid',
    'xpidt': 'xpidt',
    'xpidtf': 'xpidtf',
    'oracle': 'xpidtvckm',
}
vary_dict = {
    'fixed': {
        'damping_constant_bounds': (4.0, 4.0),
        'spring_stiffness_bounds': (2.0, 2.0),
        'mass_bounds': (20.0, 20.0),
    },
    'small': {
        'damping_constant_bounds': (3.5, 5.5),
        'spring_stiffness_bounds': (1.75, 3.0),
        'mass_bounds': (17.5, 40.0),
    },
    'med': {
        'damping_constant_bounds': (3.0, 7.0),
        'spring_stiffness_bounds': (1.25, 4.0),
        'mass_bounds': (15.0, 60.0),
    },
    'large': {
        'damping_constant_bounds': (2.0, 10.0),
        'spring_stiffness_bounds': (0.5, 6.0),
        'mass_bounds': (10.0, 100.0),
    },
}
for obk, obv in obs_dict.items():
    for vak, vav in vary_dict.items():
        kwargs = dict(vav)
        kwargs['observations'] = obv
        kwargs['action_is_change'] = 'f' in obv
        register(
            id=f'msd-{vak}-{obk}-v0',
            entry_point=MassSpringDamperEnv,
            kwargs=kwargs,
        )
        register(
            id=f'dmsd-{vak}-{obk}-v0',
            entry_point=DoubleMassSpringDamperEnv,
            kwargs=kwargs,
        )

for obk, obv in obs_dict.items():
    for vak, vav in vary_dict.items():
        kwargs = dict(vav)
        kwargs['observations'] = obv
        kwargs['action_is_change'] = 'f' in obv
        kwargs['i_horizon'] = 32
        register(
            id=f'msd-{vak}-{obk}-i32-v0',
            entry_point=MassSpringDamperEnv,
            kwargs=kwargs,
        )
        register(
            id=f'dmsd-{vak}-{obk}-i32-v0',
            entry_point=DoubleMassSpringDamperEnv,
            kwargs=kwargs,
        )

for obk, obv in obs_dict.items():
    for vak, vav in vary_dict.items():
        kwargs = dict(vav)
        kwargs['observations'] = obv
        kwargs['action_is_change'] = 'f' in obv
        kwargs['i_horizon'] = 16
        register(
            id=f'msd-{vak}-{obk}-i16-v0',
            entry_point=MassSpringDamperEnv,
            kwargs=kwargs,
        )
        register(
            id=f'dmsd-{vak}-{obk}-i16-v0',
            entry_point=DoubleMassSpringDamperEnv,
            kwargs=kwargs,
        )

###########################################################################
# Register fusion environments.
###########################################################################
register(
    id='fusion-sim-xtf-v0',
    entry_point=ToyFusionEnv,
    kwargs={'include_pid_in_obs': False},
)
register(
    id='fusion-sim-xpidtf-v0',
    entry_point=ToyFusionEnv,
    kwargs={'include_pid_in_obs': True},
)
register(
    id='fusion-sim-pid-v0',
    entry_point=ToyFusionEnv,
    kwargs={'include_pid_in_obs': True,
            'obs_is_pid_only': True,
            'action_is_change': True},
)
register(
    id='bnrot-sim-xtf-v0',
    entry_point=BetanRotEnv,
    kwargs={'include_pid_in_obs': False},
)
register(
    id='bnrot-sim-xpidtf-v0',
    entry_point=BetanRotEnv,
    kwargs={'include_pid_in_obs': True},
)
register(
    id='bnrot-sim-pid-v0',
    entry_point=BetanRotEnv,
    kwargs={'include_pid_in_obs': True,
            'obs_is_pid_only': True,
            'action_is_change': True},
)

###########################################################################
# Register nav environments.
###########################################################################
obs_dict = {a: a for a in ['x', 'xt', 'xtf', 'xpid', 'xpidt', 'pid']}
vary_dict = {
    'sim': {
        'mass_bounds': (15.0, 25.0),
        'friction_bounds': (0.0, 0.0),
        'skfdiff_bounds': (0.0, 0.0),
    },
    'real': {
        'mass_bounds': (5.0, 35.0),
        'friction_bounds': (0.05, 0.25),
        'skfdiff_bounds': (0.25, 0.75),
    },
}
for obk, obv in obs_dict.items():
    for vak, vav in vary_dict.items():
        kwargs = dict(vav)
        kwargs['observations'] = obv
        register(
            id=f'unav-{vak}-{obk}-v0',
            entry_point=UNavigation,
            kwargs=kwargs,
        )
        register(
            id=f'nav-{vak}-{obk}-v0',
            entry_point=Navigation,
            kwargs=kwargs,
        )

# Register all the partially observable pybullet environments.
# Taken from https://github.com/twni2016/pomdp-baselines
"""
The observation space can be divided into several parts:
np.concatenate(
[
    z - self.initial_z, # pos
    np.sin(angle_to_target), # pos
    np.cos(angle_to_target), # pos
    0.3 * vx, # vel
    0.3 * vy, # vel
    0.3 * vz, # vel
    r, # pos
    p # pos
], # above are 8 dims
[j], # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
[self.feet_contact], # depends on foot_list, belongs to pos
])
"""
register(
    "HopperBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=list(range(15)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "HopperBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14],  # one foot
    ),  # pos
    max_episode_steps=1000,
)

register(
    "HopperBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HopperBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "WalkerBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=list(range(22)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "WalkerBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21],  # 2 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "WalkerBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("Walker2DBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "AntBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=list(range(28)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "AntBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=[
            0,
            1,
            2,
            6,
            7,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            25,
            26,
            27,
        ],  # 4 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "AntBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("AntBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23],
    ),  # vel
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-F-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=list(range(26)),
    ),  # full obs
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-P-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=[
            0,
            1,
            2,
            6,
            7,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            21,
            22,
            23,
            24,
            25,
        ],  # 6 feet
    ),  # pos
    max_episode_steps=1000,
)

register(
    "HalfCheetahBLT-V-v0",
    entry_point=POMDPWrapper,
    kwargs=dict(
        env=gym.make("HalfCheetahBulletEnv-v0"),
        partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
    ),  # vel
    max_episode_steps=1000,
)
