"""
Navigation on U shaped maze..

Author: Ian Char
Date: March 4, 2023
"""
from typing import Any, Dict, Tuple, Optional
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


FULL_OBS = ('X', 'P', 'I', 'D', 'T', 'V', 'M', 'K', 'F')
CELL_SIZE = 5
REW_SCALING = 1 / (2 * CELL_SIZE)
OBS_MAP = {
    'X': (0, 2),
    'P': (2, 4),
    'I': (4, 6),
    'D': (6, 8),
    'T': (8, 10),
    'V': (10, 12),
    'M': (12, 13),
    'K': (13, 15),
    'F': (15, 17),
}


class Navigation(gym.Env):
    """Environment of navigation.

    Observation Space:
    Action Space: Force 2D
    Reward: -abs(P)
    """

    def __init__(
        self,
        observations: Tuple[str] = FULL_OBS,
        mass_bounds: Tuple[float, float] = (20.0, 20.0),
        friction_bounds: Tuple[float, float] = (0.0, 0.0),
        skfdiff_bounds: Tuple[float, float] = (0.0, 0.0),
        dt: float = 0.1,
        # Deprecated. This actually has no influence. See sample_targets
        target_bounds: Tuple[float, float] = ((1.0, 1.0), (1.0, 1.0)),
        force_bounds: Tuple[float] = (-10, 10),
        reset_task_on_reset: bool = True,
        max_episode_steps: int = 100,
        force_penalty_coef: float = 0.1,
        max_velmag_bound: float = 1.0,
    ):
        """Constructor.

        Args:
            observations: The possible observations. The includes:
                X: The position.
                P: The Error.
                I: Integral of the error.
                D: The difference in the error.
                T: The target.
                V: The forward velocity.
                M: The mass.
                F: The friction.
            mass_bounds: Bounds for drawing the mass.
            friction_bounds: Bounds for drawing the mass.
            skfratio_bounds: Bound in multiplier for how much smaller kinetic friction
                coefficient is vs static friction ratio.
            force_penalty_coef: Force penalty on the reward.
        """
        assert len(observations) > 0 and len(observations) < 11
        self.observations = [o.lower() for o in observations]
        self.obs_dim = np.sum([OBS_MAP[o.upper()][1] - OBS_MAP[o.upper()][0]
                               for o in observations])
        self.observation_space = gym.spaces.Box(-np.ones(self.obs_dim),
                                                np.ones(self.obs_dim))
        self.action_space = gym.spaces.Box(-1 * np.ones(2), np.ones(2))
        self._mass_bounds = mass_bounds
        self._friction_bounds = friction_bounds
        self._skfdiff_bounds = skfdiff_bounds
        self._dt = dt
        self._force_bounds = force_bounds
        self._force_penalty_coef = force_penalty_coef
        self._x_target_bounds = target_bounds[0]
        self._y_target_bounds = target_bounds[1]
        self._x_start_bounds = (0, 0)
        self._y_start_bounds = (0, 0)
        self._max_velmag_bound = max_velmag_bound
        self.state = None
        self.reset_task()
        self._dynamics_mat = None
        self._max_episode_steps = max_episode_steps
        self.t = 0

    def reset(
        self,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reset the system. If parameters are provided set them.

        Args:
            task: The task as [damping, stiffness, mass] with shape (3).
            targets: The target position to hit with shape (max_horizon)

        Returns:
            The observation.
        """
        self.t = 0
        self._last_act = np.zeros(2)
        self.reset_task(task)
        # Right now targets are
        self.target = targets
        if self.target is None:
            self.target = self.sample_targets(1)[0]
        if start is None:
            self.state = self.sample_starts(1)[0]
        else:
            self.state = start
        self.iterm, dterm = [np.zeros(2) for _ in range(2)]
        return self._form_observation(np.concatenate([
            self.state[:2],
            self.target - self.state[:2],
            self.iterm,
            dterm,
            self.target,
            -self.state[2:],
            self.get_task(),
            self._last_act
        ]))

    def sample_starts(self, n_starts):
        starts = np.concatenate([
            np.array([random.uniform(*self._x_start_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
            np.array([random.uniform(*self._y_start_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
            np.zeros((n_starts, 2)),
        ], axis=1)
        return starts

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([
                random.uniform(*self._friction_bounds),
                random.uniform(*self._skfdiff_bounds),
                random.uniform(*self._mass_bounds)])
            for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._static_friction = task[0]
        self._kinetic_friction = task[0] * task[1]
        self._mass = task[2]

    def get_task(self):
        if (np.isclose(self._kinetic_friction, 0.0)
                and np.isclose(self._static_friction, 0.0)):
            return np.array([0, 0, self._mass])
        return np.array([
            self._static_friction,
            self._kinetic_friction / self._static_friction,
            self._mass,
        ])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """"Take a step in the environment.

        Args:
            action: The force to applied.

        Returns:
            The next state, the reward, whether it is terminal and info dict.
        """
        self.t += 1
        # Get the external force.
        clipped_act = np.clip(action, -1, 1)
        self._last_act = clipped_act
        ext_force = (clipped_act + 1) / 2
        ext_force = ext_force * (self._force_bounds[1] - self._force_bounds[0])
        ext_force += self._force_bounds
        # Calculate the total force.
        static_friction = np.clip(
            np.abs(np.sum(self.state[2:])) < 1e-6  # Only applies if stationary.
            * self._mass * self._static_friction * -1 * np.sign(ext_force),
            -1 * np.abs(ext_force),
            np.abs(ext_force)
        )
        kinetic_friction = (
            np.abs(np.sum(self.state[2:])) >= 1e-6  # Only applies if moving.
            * self._mass * self._kinetic_friction * -1 * np.sign(self.state[2:]),
        )
        total_force = static_friction + kinetic_friction + ext_force
        # Calculate the next state.
        next_state = np.concatenate([
            self.state[:2] + self._dt * self.state[2:],
            np.clip(self.state[2:] + total_force / self._mass,
                    -self._max_velmag_bound,
                    self._max_velmag_bound).flatten(),
        ])
        # Check for collisions and account for this accordingly. This section is
        # imperfect for sure but should be OK for small enough dt and low enough vel.
        # Hit left or right boundary.
        if next_state[0] < -CELL_SIZE or next_state[0] > CELL_SIZE:
            next_state[2] *= -1
        # Hit bottom or top boundary.
        if next_state[1] < -CELL_SIZE or next_state[1] > CELL_SIZE:
            next_state[3] *= -1
        next_state[:2] = np.clip(
            next_state[:2],
            -1 * np.array([CELL_SIZE, CELL_SIZE]),
            np.array([CELL_SIZE, CELL_SIZE])
        )
        # Compute observations and rewards.
        err = next_state[:2] - self.target
        self.iterm += err
        obs = self._form_observation(np.concatenate([
            next_state[:2],
            self.target - self.state[:2],
            self.iterm,
            -self.state[2:],
            self.target,
            -next_state[2:],
            self.get_task(),
            self._last_act,
        ]))
        rew = (-1 * REW_SCALING * np.sum(np.abs(self.target - self.state[:2]))
               - self._force_penalty_coef * np.sum(np.abs(self._last_act)))
        self.state = next_state
        return obs, rew, False, {'target': self.target}

    def rollout(
        self,
        policy,
        num_rollouts: int,
        horizon: int,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Do many rollouts at once following a policy. This should be much faster
           because we can vectorize the dynamics update.

        Args:
            policy: The policy to use for collections.
            num_rollouts: The number of rollouts to do.
            horizon: The length of the rollout.
            tasks: The tasks to roll forward with with shape (num rollouts, 3).
            targets: The tasks to roll forward with with shape
                (num rollouts, max_horizon).
            start_states: states to start at with shape (num_rollouts, 2)

        Returns dictionary with:
            - 'observations' (num_collects, horizon + 1, obs_dim)
            - 'actions' (num_collects, horizon, act_dim)
            - 'rewards' (num_collects, horizon)
            - 'terminals' (num_collects, horizon)
            - 'targets' (num_collects, horizon)
            - 'logpi': The log probabilities of actions (num_collects, horizon).
            - 'env_info': Information about the draws of the environment with
                shape (num_collects,).
        """
        policy.reset()
        # Draw system parameters and organize them into tensor.
        if task is None:
            task = self.sample_tasks(num_rollouts)
        assert len(task) == num_rollouts
        sfrictions = task[:, [0]]
        kfrictions = task[:, [0]] * task[:, [1]]
        masses = task[:, [2]]
        # Draw targets.
        if targets is None:
            targets = self.sample_targets(num_rollouts)
        assert len(targets) == num_rollouts
        # Initialize all of the data structures.
        obs = np.zeros((num_rollouts, horizon + 1, self.obs_dim))
        states = np.zeros((num_rollouts, horizon + 1, 4))
        acts = np.zeros((num_rollouts, horizon, 2))
        act_accum = np.zeros((num_rollouts, horizon + 1, 2))
        log_pis = np.zeros((num_rollouts, horizon))
        rewards = np.zeros((num_rollouts, horizon))
        terminals = np.full((num_rollouts, horizon), False)
        errs = np.zeros((num_rollouts, horizon + 1, 2))
        # Get initial states and observations.
        if start is None:
            start = self.sample_starts(num_rollouts)
        assert start.shape == (num_rollouts, 4)
        states[:, 0] = start
        # Get initial states and observations.
        if start is None:
            start = self.sample_starts(num_rollouts)
        assert start.shape == (num_rollouts, 4)
        states[:, 0] = start
        dterm = np.zeros((num_rollouts, 2))
        obs[:, 0] = self._form_observation(np.concatenate([
            states[:, 0, :2],
            (targets - states[:, 0, :2]),
            np.zeros((num_rollouts, 2)),
            dterm,
            targets,
            -states[:, 0, 2:],
            task,
            np.zeros((num_rollouts, 2)),
        ], axis=1))
        for h in range(horizon):
            ob = obs[:, h]
            pi_act, log_pi = policy.get_actions(ob)
            acts[:, h] = pi_act
            log_pis[:, h] = log_pi
            clipped_act = np.clip(pi_act, -1, 1)
            act_accum[:, h + 1] = clipped_act
            action = (clipped_act + 1) / 2
            action = action * (self._force_bounds[1] - self._force_bounds[0])
            action += self._force_bounds[0]
            # Get the total force.
            is_stat = (np.sum(np.abs(states[:, h, 2:]), axis=1) < 1e-6).reshape(-1, 1)
            static_friction = np.clip(
                is_stat * sfrictions * masses * -1 * np.sign(action),
                -1 * np.abs(action),
                np.abs(action)
            )
            kinetic_friction = ((1 - is_stat) * kfrictions * masses
                                * -1 * np.sign(states[:, h, 2:]))
            accelerations = (action + static_friction + kinetic_friction) / masses
            # Calculate the next state.
            states[:, h + 1, :2] = states[:, h, :2] + self._dt * states[:, h, 2:]
            states[:, h + 1, 2:] = np.clip(
                states[:, h, 2:] + accelerations,
                -self._max_velmag_bound,
                self._max_velmag_bound
            )
            # Check for collisions.
            # Hit left or right boundary.
            violations = np.argwhere(np.logical_or(
                states[:, h + 1, 0] < -CELL_SIZE,
                states[:, h + 1, 0] > CELL_SIZE
            ))
            states[violations, h + 1, 2] *= -1
            # Hit bottom or top boundary.
            violations = np.argwhere(np.logical_or(
                states[:, h + 1, 1] < -CELL_SIZE,
                states[:, h + 1, 1] > CELL_SIZE
            ))
            states[violations, h + 1, 3] *= -1
            states[:, h + 1, :2] = np.clip(
                states[:, h + 1, :2],
                -1 * np.array([CELL_SIZE, CELL_SIZE]),
                np.array([CELL_SIZE, CELL_SIZE]),
            )
            # Form the observations.
            errs[:, h + 1] = states[:, h + 1, :2] - targets
            obs[:, h + 1] = self._form_observation(np.concatenate([
                states[:, h + 1, :2],
                -errs[:, h + 1],
                -np.sum(errs[:, :h + 2], axis=1) * self._dt,
                -states[:, h, 2:],
                targets,
                -states[:, h + 1, 2:],
                task,
                clipped_act,
            ], axis=1))
            rewards[:, h] = (
                np.sum(-REW_SCALING * np.abs(errs[:, h + 1]), axis=1)
                - self._force_penalty_coef * np.sum(np.abs(clipped_act), axis=1))
            if hasattr(policy, 'get_reward_feedbck'):
                policy.get_reward_feedback(rewards[:, h])
        return {
            'observations': obs[:, :-1, :],
            'next_observations': obs[:, 1:, :],
            'actions': acts,
            'rewards': rewards[..., np.newaxis],
            'terminals': terminals[..., np.newaxis],
            'targets': targets,
            'logpi': log_pis,
        }

    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, 2, max horizon).
        """
        return np.concatenate([
            (np.random.randint(2, size=(num_target_trajs, 1)) * 2 - 1)
            * np.random.uniform(CELL_SIZE / 3, CELL_SIZE, size=(num_target_trajs, 1))
            for _ in range(2)
        ], axis=1)

    def _form_observation(self, full_obs):
        """Form observation based on what self.observations"""
        idx_list = []
        for ob_name in self.observations:
            idx_list += [i for i in range(*OBS_MAP[ob_name.upper()])]
        if len(full_obs.shape) == 1:
            return full_obs[idx_list]
        return full_obs[:, idx_list]

    def plot_paths(self, rollouts, colors=('blue', 'red', 'green', 'orange', 'purple')):
        plt.style.use('seaborn')
        if len(rollouts['observations']) > 5:
            raise ValueError('Cannot handle that many rollouts')
        num_plots = len(self.observations) - int('t' in self.observations)
        fig, axs = plt.subplots(num_plots, 2)
        curr_idx = 0
        for name in FULL_OBS:
            if name.lower() in self.observations and name != 'T':
                for col in range(2):
                    axs[curr_idx, col].set_title(name.upper() + str(col + 1))
                    axs[curr_idx, col].axhline(0, ls=':', color='black')
                curr_idx += 1
        for cidx in range(len(rollouts['observations'])):
            curr_name_idx = 0
            pidx = 0
            aidx = 0
            while pidx < num_plots and aidx < len(self.observations) * 2:
                while FULL_OBS[curr_name_idx].lower() not in self.observations:
                    curr_name_idx += 1
                if FULL_OBS[curr_name_idx] != 'T':
                    axs[pidx, 0].plot(
                        rollouts['observations'][cidx, :, aidx].flatten(),
                        color=colors[cidx])
                    axs[pidx, 1].plot(
                        rollouts['observations'][cidx, :, aidx + 1].flatten(),
                        color=colors[cidx])
                    pidx += 1
                curr_name_idx += 1
                aidx += 2
            if 'x' in self.observations:
                axs[0, 0].axhline(
                    rollouts['targets'][cidx, 0], color=colors[cidx], ls='--',
                    label=f'{np.sum(rollouts["rewards"][cidx]):0.2f}')
                axs[0, 1].axhline(
                    rollouts['targets'][cidx, 1], color=colors[cidx], ls='--')
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()
