"""
System with two weights and three springs.

Author: Ian Char
Date: January 8, 2022
"""
from collections import deque
from typing import Any, Dict, Tuple, Optional
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


FULL_OBS = ('X', 'P', 'I', 'D', 'T', 'V', 'C', 'K', 'M', 'F')
OBS_MAP = {
    'X': (0, 2),
    'P': (2, 4),
    'I': (4, 6),
    'D': (6, 8),
    'T': (8, 10),
    'V': (10, 12),
    'C': (12, 15),
    'K': (15, 18),
    'M': (18, 20),
    'F': (20, 22),
}


class DoubleMassSpringDamperEnv(gym.Env):
    """Environment of two masses back to back in series with a wall on either side.

    Observation Space: Maximum: (P, I, D) 3D
    Action Space: Force 1D
    Reward: -abs(P)
    """

    def __init__(
        self,
        observations: Tuple[str] = FULL_OBS,
        damping_constant_bounds: Tuple[float, float] = (4.0, 4.0),
        spring_stiffness_bounds: Tuple[float, float] = (2.0, 2.0),
        mass_bounds: Tuple[float, float] = (20.0, 20.0),
        target_bounds: Tuple[float, float] = (-1.5, 1.5),
        dt: float = 0.2,
        force_bounds: Tuple[float] = (-30, 30),
        reset_task_on_reset: bool = True,
        max_targets_per_ep: int = 2,
        action_is_change: bool = False,
        first_d_is_p: bool = False,
        i_horizon: Optional[int] = None,
        max_episode_steps: int = 100,
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
                C: The damping constant.
                K: The spring constant.
                M: The mass constant.
                F: The current force setting.
            damping_constant_bounds: Bounds for drawing damping constant for the
            system.
            spring_stiffness_bounds: Bounds for drawing spring stiffnesses.
            mass_bounds: Bounds for drawing the mass.
            target_bounds: Bounds for the possible targets.
            dt: The discrete time step to take.
            force_bounds: Bounds on the force that can be applied.
            max_targets_per_ep: The maximum number of targets that can be in any
                one task.
            action_is_change: Whether the action should be the change in force.
                The force is in term of the standardized, clipped [-1, 1] amount.
            first_d_is_p: Whether the first difference should just be the error.
            i_horizon: How far the lookback should be for computing the I term.
            max_episode_steps: Length of an episode.
        """
        assert len(observations) > 0 and len(observations) < 11
        self.observations = tuple([o.lower() for o in observations])
        self.obs_dim = np.sum([OBS_MAP[o.upper()][1] - OBS_MAP[o.upper()][0]
                               for o in observations])
        self.observation_space = gym.spaces.Box(-np.ones(self.obs_dim),
                                                np.ones(self.obs_dim))
        self.action_space = gym.spaces.Box(-1 * np.ones(2), np.ones(2))
        self.task_dim = 2 + 2 * 3
        self.max_targets_per_ep = max_targets_per_ep
        self._damping_bounds = damping_constant_bounds
        self._stiffness_bounds = spring_stiffness_bounds
        self._mass_bounds = mass_bounds
        self._target_bounds = target_bounds
        self._dt = dt
        self._force_bounds = force_bounds
        self._start_posn_bounds = (-0.25, 0.25)
        self._start_vel_bounds = (-0.1, 0.1)
        self._action_is_change = action_is_change
        self._first_d_is_p = first_d_is_p
        if i_horizon is not None:
            if i_horizon < 1:
                raise ValueError(f'Horizon must be at least 1, received {i_horizon}')
        self._i_horizon = i_horizon
        self._err_hist = deque(maxlen=self._i_horizon)
        self.state = None
        self.reset_task()
        self._dynamics_mat = None
        self._max_episode_steps = max_episode_steps
        self._reset_task_on_reset = reset_task_on_reset
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
        self._err_hist = deque(maxlen=self._i_horizon)
        if self._reset_task_on_reset:
            self.reset_task(task)
        self._targets = targets
        if self._targets is None:
            self._targets = self.sample_targets(1)[0]
        self._dynamics_mat = np.array([
            [1, 0, self._dt, 0],  # First position update.
            [0, 1, 0, self._dt],  # Second position update.
            [-self._dt * (self._k1 + self._k2) / self._m1,  # First vel update.
             self._dt * self._k2 / self._m1,
             1 - self._dt * (self._d1 + self._d2) / self._m1,
             -self._dt * self._d2 / self._m1],
            [self._dt * self._k2 / self._m2,  # Seoncd vel update.
             -self._dt * (self._k2 + self._k3) / self._m2,
             self._dt * self._d2 / self._m2,
             1 - self._dt * (self._d2 + self._d3) / self._m2]])
        if start is None:
            self.state = self.sample_starts(1)[0]
        else:
            self.state = start
        dterm = self.target - self.state[:2] if self._first_d_is_p else np.zeros(2)
        return self._form_observation(np.concatenate([
            self.state[:2],
            self.target - self.state[:2],
            self.iterm,
            dterm,
            self.target,
            -self.state[2:],
            self.get_task(),
            self._last_act,
        ]))

    def sample_starts(self, n_starts):
        starts = np.concatenate([
            np.array([random.uniform(*self._start_posn_bounds)
                      for _ in range(2 * n_starts)]).reshape(-1, 2),
            np.array([random.uniform(*self._start_vel_bounds)
                      for _ in range(2 * n_starts)]).reshape(-1, 2),
        ], axis=1)
        return starts

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([
                random.uniform(*self._damping_bounds),
                random.uniform(*self._damping_bounds),
                random.uniform(*self._damping_bounds),
                random.uniform(*self._stiffness_bounds),
                random.uniform(*self._stiffness_bounds),
                random.uniform(*self._stiffness_bounds),
                random.uniform(*self._mass_bounds),
                random.uniform(*self._mass_bounds)])
            for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._d1 = task[0]
        self._d2 = task[1]
        self._d3 = task[2]
        self._k1 = task[3]
        self._k2 = task[4]
        self._k3 = task[5]
        self._m1 = task[6]
        self._m2 = task[7]

    def get_task(self):
        return np.array([
            self._d1,
            self._d2,
            self._d3,
            self._k1,
            self._k2,
            self._k3,
            self._m1,
            self._m2,
        ])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def target(self):
        return self._targets[np.min([self.t, self._max_episode_steps - 1])]

    @property
    def iterm(self) -> float:
        if len(self._err_hist):
            return -np.sum(np.array(self._err_hist), axis=0) * self._dt
        return np.zeros(2)

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
        if self._action_is_change:
            clipped_act = np.clip(self._last_act + action, -1, 1)
        else:
            clipped_act = np.clip(action, -1, 1)
        self._last_act = clipped_act
        action = (clipped_act + 1) / 2
        action = action * (self._force_bounds[1] - self._force_bounds[0])
        action += self._force_bounds[0]
        prev_change = self.state[2:]
        self.state = (self._dynamics_mat @ self.state.reshape(-1, 1)).flatten()
        self.state[2:] += self._dt * np.array([1 / self._m1, 1 / self._m2]) * action
        err = self.state[:2] - self.target
        self._err_hist.append(err)
        obs = self._form_observation(np.concatenate([
            self.state[:2],
            self.target - self.state[:2],
            self.iterm,
            -prev_change,
            self.target,
            -self.state[2:],
            self.get_task(),
            self._last_act,
        ]))
        return obs, np.sum(-np.abs(err)), False, {'target': self.target}

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
        dynamics_tensor = np.array([
            np.array([
                [1, 0, self._dt, 0],  # First position update.
                [0, 1, 0, self._dt],  # Second position update.
                [-self._dt * (ta[3] + ta[4]) / ta[6],  # First vel update.
                 self._dt * ta[4] / ta[6],
                 1 - self._dt * (ta[0] + ta[1]) / ta[6],
                 -self._dt * ta[1] / ta[6]],
                [self._dt * ta[4] / ta[7],  # Seoncd vel update.
                 -self._dt * (ta[4] + ta[5]) / ta[7],
                 self._dt * ta[1] / ta[7],
                 1 - self._dt * (ta[1] + ta[2]) / ta[7]]])
            for ta in task
        ])
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
        dterm = ((targets[:, 0] - states[:, 0, :2]) if self._first_d_is_p
                 else np.zeros((num_rollouts, 2)))
        obs[:, 0] = self._form_observation(np.concatenate([
            states[:, 0, :2],
            (targets[:, 0] - states[:, 0, :2]),
            np.zeros((num_rollouts, 2)),
            dterm,
            targets[:, 0],
            -states[:, 0, 2:],
            task,
            np.zeros((num_rollouts, 2)),
        ], axis=1))
        for h in range(horizon):
            ob = obs[:, h]
            pi_act, log_pi = policy.get_actions(ob)
            acts[:, h] = pi_act
            log_pis[:, h] = log_pi
            if not self._action_is_change:
                clipped_act = np.clip(pi_act, -1, 1)
            else:
                clipped_act = np.clip(act_accum[:, h] + pi_act, -1, 1)
            act_accum[:, h + 1] = clipped_act
            action = (clipped_act + 1) / 2
            action = action * (self._force_bounds[1] - self._force_bounds[0])
            action += self._force_bounds[0]
            states[:, h + 1] =\
                ((dynamics_tensor @ states[:, h][..., np.newaxis])
                 .reshape(num_rollouts, 4))
            states[:, h + 1, 2:] += self._dt / task[:, -2:] * action
            curr_targets = targets[:, np.min([h + 1, targets.shape[1] - 1])]
            errs[:, h + 1] = states[:, h + 1, :2] - curr_targets
            if self._i_horizon is not None:
                err_lookback = max(h + 2 - self._i_horizon, 0)
            else:
                err_lookback = 0
            obs[:, h + 1] = self._form_observation(np.concatenate([
                states[:, h + 1, :2],
                -errs[:, h + 1],
                -np.sum(errs[:, err_lookback:h + 2], axis=1) * self._dt,
                -states[:, h, 2:],
                curr_targets,
                -states[:, h + 1, 2:],
                task,
                clipped_act,
            ], axis=1))
            rewards[:, h] = np.sum(-np.abs(errs[:, h + 1]), axis=1)
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
        targets_to_return = []
        for _ in range(num_target_trajs):
            curr_target = []
            for _ in range(2):
                num_targets = random.randint(1, self.max_targets_per_ep)
                targets = np.array([])
                for tnum in range(num_targets):
                    if tnum == num_targets - 1:
                        targets = np.append(
                            targets,
                            np.ones(self._max_episode_steps - len(targets))
                            * random.uniform(*self._target_bounds),
                        )
                    else:
                        targets = np.append(
                            targets,
                            np.ones(self._max_episode_steps // num_targets)
                            * random.uniform(*self._target_bounds),
                        )
                curr_target.append(targets)
            targets_to_return.append(np.array(curr_target).T)
        return np.array(targets_to_return)

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
        obs_set = set(['x', 'f'])
        if not obs_set.issubset(self.observations):
            raise ValueError('Cannot visualize with this observation subset.')
        if len(rollouts['observations']) > 5:
            raise ValueError('Cannot handle that many rollouts')
        num_plots = len(obs_set) - int('t' in obs_set)
        fig, axs = plt.subplots(num_plots, 2)
        curr_idx = 0
        for name in FULL_OBS:
            if name.lower() in obs_set and name != 'T':
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
            if 'x' in obs_set:
                axs[0, 0].plot(
                    rollouts['targets'][cidx, :, 0], color=colors[cidx], ls='--',
                    label=f'{np.sum(rollouts["rewards"][cidx]):0.2f}')
                axs[0, 1].plot(
                    rollouts['targets'][cidx, :, 1], color=colors[cidx], ls='--')
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()
