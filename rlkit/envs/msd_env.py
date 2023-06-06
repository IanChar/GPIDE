"""
Mass-spring-damper system environment.

Code based on: https://www.halvorsen.blog/documents/programming/python/resources
/powerpoints/Mass-Spring-Damper%20System%20with%20Python.pdf

Author: Ian Char
Date: September 5, 2022
"""
from collections import deque
from typing import Any, Dict, Tuple, Optional
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


FULL_OBS = ('X', 'P', 'I', 'D', 'T', 'V', 'C', 'K', 'M', 'F')


class MassSpringDamperEnv(gym.Env):
    """Environment of the classic mass spring damper system.

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
        force_bounds: Tuple[float] = (-10, 10),
        reset_task_on_reset: bool = True,
        max_targets_per_ep: int = 1,
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
        """
        assert len(observations) > 0 and len(observations) < 11
        self.observations = tuple([o.lower() for o in observations])
        self.observation_space = gym.spaces.Box(-np.ones(len(observations)),
                                                np.ones(len(observations)))
        self.action_space = gym.spaces.Box(-1 * np.ones(1), np.ones(1))
        self.task_dim = 3
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
        self._last_act = 0.0
        self._err_hist = deque(maxlen=self._i_horizon)
        if self._reset_task_on_reset:
            self.reset_task(task)
        self._targets = targets
        if self._targets is None:
            self._targets = self.sample_targets(1)[0]
        self._dynamics_mat = np.array([
            [1, self._dt],
            [-self._dt * self._stiffness / self._mass,
             (1 - self._dt * self._damping / self._mass)]])
        if start is None:
            self.state = self.sample_starts(1)[0]
        else:
            self.state = start
        dterm = self.target - self.state[0] if self._first_d_is_p else 0
        return self._form_observation(np.array([
            self.state[0],
            self.target - self.state[0],
            -np.sum(self._err_hist) * self._dt,
            dterm,
            self.target,
            -self.state[1],
            self._damping,
            self._stiffness,
            self._mass,
            self._last_act,
        ]))

    def sample_starts(self, n_starts):
        return np.concatenate([
            np.array([random.uniform(*self._start_posn_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
            np.array([random.uniform(*self._start_vel_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
        ], axis=1)

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([random.uniform(*self._damping_bounds),
                                    random.uniform(*self._stiffness_bounds),
                                    random.uniform(*self._mass_bounds)])
                          for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._damping = task[0]
        self._stiffness = task[1]
        self._mass = task[2]

    def get_task(self):
        return np.array([self._damping, self._stiffness, self._mass])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def target(self):
        return self._targets[np.min([self.t, self._max_episode_steps - 1])]

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
            clipped_act = np.clip(float(action), -1, 1)
        self._last_act = clipped_act
        action = (clipped_act + 1) / 2
        action = action * (self._force_bounds[1] - self._force_bounds[0])
        action += self._force_bounds[0]
        prev_change = self.state[1]
        self.state = (self._dynamics_mat @ self.state.reshape(-1, 1)).flatten()
        self.state[1] += self._dt / self._mass * float(action)
        err = self.state[0] - self.target
        self._err_hist.append(err)
        obs = self._form_observation(np.array([
            self.state[0],
            -err,
            -np.sum(self._err_hist) * self._dt,
            -prev_change,
            self.target,
            -self.state[1],
            self._damping,
            self._stiffness,
            self._mass,
            float(self._last_act),
        ]))
        return obs, -np.abs(err), False, {'target': self.target}

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
            np.array([[1, self._dt],
                      [-self._dt * ta[1] / ta[2],
                       1 - self._dt * ta[0] / ta[2]]])
            for ta in task
        ])
        # Draw targets.
        if targets is None:
            targets = self.sample_targets(num_rollouts)
        assert len(targets) == num_rollouts
        # Initialize all of the data structures.
        obs = np.zeros((num_rollouts, horizon + 1, len(self.observations)))
        states = np.zeros((num_rollouts, horizon + 1, 2))
        acts = np.zeros((num_rollouts, horizon, 1))
        act_accum = np.zeros((num_rollouts, horizon + 1, 1))
        log_pis = np.zeros((num_rollouts, horizon))
        rewards = np.zeros((num_rollouts, horizon))
        terminals = np.full((num_rollouts, horizon), False)
        errs = np.zeros((num_rollouts, horizon + 1))
        # Get initial states and observations.
        if start is None:
            start = self.sample_starts(num_rollouts)
        assert start.shape == (num_rollouts, 2)
        states[:, 0] = start
        dterm = ((targets[:, 0] - states[:, 0, 0]).reshape(-1, 1) if self._first_d_is_p
                 else np.zeros((num_rollouts, 1)))
        obs[:, 0] = self._form_observation(np.concatenate([
            states[:, 0, 0].reshape(-1, 1),
            (targets[:, 0] - states[:, 0, 0]).reshape(-1, 1),
            np.zeros((num_rollouts, 1)),
            dterm,
            targets[:, 0].reshape(-1, 1),
            -states[:, 0, 1].reshape(-1, 1),
            task,
            np.zeros((num_rollouts, 1)),
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
                 .reshape(num_rollouts, 2))
            states[:, h + 1, 1] += self._dt / task[:, -1] * action.flatten()
            curr_targets = targets[:, np.min([h + 1, targets.shape[1] - 1])]
            errs[:, h + 1] = states[:, h + 1, 0] - curr_targets
            if self._i_horizon is not None:
                err_lookback = max(h + 2 - self._i_horizon, 0)
            else:
                err_lookback = 0
            obs[:, h + 1] = self._form_observation(np.concatenate([
                states[:, h + 1, 0].reshape(-1, 1),
                -errs[:, h + 1].reshape(-1, 1),
                -np.sum(errs[:, err_lookback:h + 2], axis=1).reshape(-1, 1) * self._dt,
                -states[:, h, 1].reshape(-1, 1),
                curr_targets.reshape(-1, 1),
                -states[:, h + 1, 1].reshape(-1, 1),
                task,
                clipped_act,
            ], axis=1))
            rewards[:, h] = -np.abs(errs[:, h + 1])
            if hasattr(policy, 'get_reward_feedback'):
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

        Returns: The target ndarray w shape (num_target_trajs, max horizon).
        """
        targets_to_return = []
        for _ in range(num_target_trajs):
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
            targets_to_return.append(targets)
        return np.array(targets_to_return)

    def _form_observation(self, full_obs):
        """Form observation based on what self.observations"""
        idx_list = [idx for idx, o in enumerate(FULL_OBS)
                    if o.lower() in self.observations]
        if len(full_obs.shape) == 1:
            return full_obs[idx_list]
        return full_obs[:, idx_list]

    def plot_paths(self, rollouts, colors=('blue', 'red', 'green', 'orange', 'purple'),
                   is_pid=False):
        plt.style.use('seaborn')
        if is_pid:
            fig, axs = plt.subplots(2, 1)
            for cidx, color in enumerate(colors[:len(rollouts['observations'])]):
                axs[0].plot(rollouts['observations'][cidx, :, 0], color=color,
                            label=f'{np.sum(rollouts["rewards"][cidx]):0.2f}')
                axs[1].plot(rollouts['actions'][cidx].flatten(), color=color)
        else:
            obs_set = set(['x', 'f'])
            if not obs_set.issubset(self.observations):
                raise ValueError('Cannot visualize with this observation subset.')
            if len(rollouts['observations']) > 5:
                raise ValueError('Cannot handle that many rollouts')
            num_plots = len(obs_set) - int('t' in obs_set)
            fig, axs = plt.subplots(num_plots, 1)
            curr_idx = 0
            for name in FULL_OBS:
                if name.lower() in obs_set and name != 'T':
                    axs[curr_idx].set_title(name.upper())
                    axs[curr_idx].axhline(0, ls=':', color='black')
                    curr_idx += 1
            for cidx in range(len(rollouts['observations'])):
                curr_name_idx = 0
                pidx = 0
                aidx = 0
                while pidx < num_plots and aidx < len(self.observations):
                    while FULL_OBS[curr_name_idx].lower() not in self.observations:
                        curr_name_idx += 1
                    if FULL_OBS[curr_name_idx] != 'T':
                        axs[pidx].plot(
                            rollouts['observations'][cidx, :, aidx].flatten(),
                            color=colors[cidx])
                        pidx += 1
                    curr_name_idx += 1
                    aidx += 1
                if 'x' in obs_set and not is_pid:
                    axs[0].plot(rollouts['targets'][cidx], color=colors[cidx], ls='--',
                                label=f'{np.sum(rollouts["rewards"][cidx]):0.2f}')
        axs[0].legend()
        plt.tight_layout()
        plt.show()
