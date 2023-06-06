"""
Environment to do both betan and rotation tracking.

Author: Ian Char
Date: March 21, 2023
"""
from typing import Any, Dict, Tuple, Optional
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


BETAN_MU = 1.7797029
BETAN_SIG = 1.2159256
ROTATION_MU = 39.580715
ROTATION_SIG = 45.99354
PINJ_MU = 5817915.52734375
PINJ_SIG = 4223408.935546875
TINJ_MU = 4.3
TINJ_SIG = 3.53
PINJ_ACT_SCALE = 0.25 * PINJ_SIG
TINJ_ACT_SCALE = 0.25 * TINJ_SIG
TRANSITION_COEF = 2e2
BETA_OBS_COEF = 5.0
DT = 0.025


class BetanRotEnv(gym.Env):
    """Environment for Betan and Rotation tracking.

    Obvervation Space: (BetaN, Rotation, Curr Power, Curr Torque,  Targets).
    Action Space: Change in the beam powers.
    Reward: -abs(P)
    """

    def __init__(
        self,
        include_pid_in_obs: bool,
        w_start_dist: Tuple[float, float] = (5e4, 2.5e3),
        dw_start_dist: Tuple[float, float] = (0, 2.5e3),
        rot_start_dist: Tuple[float, float] = (40.0, 5.0),
        drot_start_dist: Tuple[float, float] = (0.0, 1.0),
        pinj_start_dist: Tuple[float, float] = (1.0e6, 1e5),
        tinj_start_dist: Tuple[float, float] = (4.5, 0.25),
        aminor_dist: Tuple[float, float] = (0.589, 0.02),
        bt_dist: Tuple[float, float] = (2.75, 0.1),
        ip_dist: Tuple[float, float] = (1e6, 1e5),
        confinement_dist: Tuple[float, float] = (0.1, 0.001),
        torquecoef_dist: Tuple[float, float] = (80.0, 0.001),
        betan_target_bounds: Tuple[float, float] = (1.75, 2.75),
        rot_target_bounds: Tuple[float, float] = (25.0, 50.0),
        pinj_bounds: Tuple[float, float] = (5e5, 1.2e7),
        tinj_bounds: Tuple[float, float] = (-1.976, 11.047),
        w_bounds: Tuple[float, float] = (0.0, 5e5),
        dw_bounds: Tuple[float, float] = (-8.5e5, 8.5e5),
        rot_bounds: Tuple[float, float] = (-25, 135),
        drot_bounds: Tuple[float, float] = (-386, 484.0),
        momentum: float = 0.5,
        obs_is_pid_only: bool = False,
        action_is_change: bool = True,
    ):
        """Constructor.

        Args:
            include_pid_in_obs: Whether the observation space should have PID
                components included.
            w_start_dist: Mean and std deviation for the W start distribution.
            pinj_start_dist: Mean and std deviation for the Pinj start distribution.
            pinj_start_dist: Mean and std deviation for the tinj start distribution.
            aminor_dist: Mean and std deviation for aminor.
            bt_dist: Mean and std deviation for bt.
            ip_dist: Mean and std deviation for ip.
            betan_target_bounds: Bounds on the betan target.
            rot_target_bounds: Bounds on the rotation target.
            pinj_bounds: Bounds on how high power can be.
            tinj_bounds: Bounds on how torque.
            w_bounds: Bounds on the stored energy.
            dw_bounds: Bounds on how fast the stored energy can change.
            rot_bounds: Bounds on rotation.
            drot_bounds: Bounds on change in rotation.
            momentum: Added momentum term in the dynamics.
            obs_is_pid_only: Whether the observations should only be pid.
            action_is_change: Whether the action is change in current power or is it
                the delta based on the midpoint of the power bounds.
        """
        super().__init__()
        self._include_pid_in_obs = include_pid_in_obs
        self._obs_is_pid_only = obs_is_pid_only
        self.observation_dim = 6 * (not self._obs_is_pid_only) + include_pid_in_obs * 6
        self.observation_space = gym.spaces.Box(
            -np.ones(self.observation_dim),
            np.ones(self.observation_dim))
        self._action_is_change = action_is_change
        self.action_space = gym.spaces.Box(-1 * np.ones(2), np.ones(2))
        self.task_dim = 3  # i.e. aminor, bt, and ip.
        self._w_start_dist = w_start_dist
        self._dw_start_dist = dw_start_dist
        self._rot_start_dist = rot_start_dist
        self._drot_start_dist = drot_start_dist
        self._pinj_start_dist = pinj_start_dist
        self._tinj_start_dist = tinj_start_dist
        self._aminor_dist = aminor_dist
        self._bt_dist = bt_dist
        self._ip_dist = ip_dist
        self._confinement_dist = confinement_dist
        self._torquecoef_dist = torquecoef_dist
        self._betan_target_bounds = betan_target_bounds
        self._rot_target_bounds = rot_target_bounds
        self._pinj_bounds = pinj_bounds
        self._tinj_bounds = tinj_bounds
        self._w_bounds = w_bounds
        self._dw_bounds = dw_bounds
        self._rot_bounds = rot_bounds
        self._drot_bounds = drot_bounds
        self._momentum = momentum
        self.state = None
        self._dt = 0.025
        self.reset_task()
        self._max_episode_steps = 100
        self._error_accum = None
        self.t = 0
        self._beam_bounder = CounterCurrentBeamBounding()

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
        self.reset_task(task)
        if targets is None:
            targets = self.sample_targets(1)[0]
        self.target = targets
        if start is None:
            start = self.sample_starts(1)[0]
        self.state = start
        self._bn_error_accum = None
        self._rot_error_accum = None
        return self._form_observation(self.state, self.target)

    def sample_starts(self, n_starts):
        start_ws = np.array([
            random.gauss(*self._w_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_dws = np.array([
            random.gauss(*self._dw_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_ps = np.array([
            np.clip(
                random.gauss(*self._pinj_start_dist),
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            )
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_rots = np.array([
            random.gauss(*self._rot_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_drots = np.array([
            random.gauss(*self._drot_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_ts = np.array([
            np.clip(
                random.gauss(*self._tinj_start_dist),
                self._tinj_bounds[0],
                self._tinj_bounds[1],
            )
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        return np.concatenate([
            start_ws,
            start_dws,
            start_rots,
            start_drots,
            start_ps,
            start_ts,
        ], axis=1)

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([random.gauss(*self._aminor_dist),
                                    random.gauss(*self._bt_dist),
                                    random.gauss(*self._ip_dist),
                                    random.gauss(*self._confinement_dist),
                                    random.gauss(*self._torquecoef_dist)])
                          for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._aminor = task[0]
        self._bt = task[1]
        self._ip = task[2]
        self._confinement = task[3]
        self._torque_coef = task[4]

    def get_task(self):
        return np.array([self._aminor, self._bt, self._ip,
                         self._confinement, self._torque_coef])

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
        if self._action_is_change:
            pinj_change = np.clip(action[0], -1, 1) * PINJ_ACT_SCALE
            next_pinj = np.clip(
                self.state[4] + pinj_change,
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            )
            tinj_change = np.clip(action[1], -1, 1) * TINJ_ACT_SCALE
            next_tinj = np.clip(
                self.state[5] + tinj_change,
                self._tinj_bounds[0],
                self._tinj_bounds[1]
            )
        else:
            midpoint = (self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
            next_pinj = np.clip(
                np.clip(action[0], -1, 1) * midpoint + midpoint,
                self.state[4] - PINJ_ACT_SCALE,
                self.state[4] + PINJ_ACT_SCALE,
            )
            midpoint = (self._tinj_bounds[1] - self._tinj_bounds[0]) / 2
            next_tinj = np.clip(
                np.clip(action[1], -1, 1) * midpoint + midpoint,
                self.state[5] - TINJ_ACT_SCALE,
                self.state[5] + TINJ_ACT_SCALE,
            )
        next_pinj, next_tinj = self._beam_bounder.bound_beams(next_pinj, next_tinj)
        next_dw = (self._momentum * self.state[1]
                   + (1 - self._momentum) * np.clip(
                          next_pinj
                          - TRANSITION_COEF * self.state[0]
                          * self._ip ** -0.93
                          * self._bt ** -0.15
                          * self.state[4] ** 0.69),
                   self._dw_bounds[0], self._dw_bounds[1])
        next_w = np.clip(
            self.state[0] + self._dt * self.state[1],
            self._w_bounds[0],
            self._w_bounds[1],
        )
        next_drot = (self._momentum * self.state[3]
                     + (1 - self._momentum) * np.clip(
                     next_tinj * self._torque_coef
                     - self.state[2] / self._confinement
                     ), self._drot_bounds[0], self._drot_bounds[1])
        next_rot = np.clip(
            self.state[2] + self._dt * self.state[3],
            self._rot_bounds[0],
            self._rot_bounds[1],
        )
        next_state = np.array([next_w, next_dw, next_rot, next_drot,
                               next_pinj, next_tinj])
        obs, rew = self._form_observation_and_rew(
            next_state, self.target, None, self.state)
        self.state = next_state
        rew = np.abs(obs[:2] - self.target)
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
        self._bn_error_accum = None
        self._rot_error_accum = None
        policy.reset()
        # Draw system parameters and organize them into tensor.
        if task is None:
            task = self.sample_tasks(num_rollouts)
        assert len(task) == num_rollouts
        # Draw targets.
        if targets is None:
            targets = self.sample_targets(num_rollouts)
        assert len(targets) == num_rollouts
        # Initialize all of the data structures.
        obs = np.zeros((num_rollouts, horizon + 1, self.observation_dim))
        states = np.zeros((num_rollouts, horizon + 1, 6))
        acts = np.zeros((num_rollouts, horizon, 2))
        log_pis = np.zeros((num_rollouts, horizon))
        rewards = np.zeros((num_rollouts, horizon))
        terminals = np.full((num_rollouts, horizon), False)
        # Get initial states and observations.
        if start is None:
            start = self.sample_starts(num_rollouts)
        assert start.shape == (num_rollouts, 6)
        states[:, 0] = start
        obs[:, 0] = self._form_observation(states[:, 0], targets, task)
        for h in range(horizon):
            ob = obs[:, h]
            state = states[:, h]
            pi_act, log_pi = policy.get_actions(ob)
            acts[:, h] = pi_act
            log_pis[:, h] = log_pi
            if self._action_is_change:
                pinj_change = np.clip(pi_act[:, 0], -1, 1) * PINJ_ACT_SCALE
                next_pinj = np.clip(
                    state[:, 4] + pinj_change.flatten(),
                    self._pinj_bounds[0],
                    self._pinj_bounds[1]
                )
                tinj_change = np.clip(pi_act[:, 1], -1, 1) * TINJ_ACT_SCALE
                next_tinj = np.clip(
                    state[:, 5] + tinj_change.flatten(),
                    self._tinj_bounds[0],
                    self._tinj_bounds[1]
                )
            else:
                bound_radius = (self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
                midpoint = bound_radius + self._pinj_bounds[0]
                next_pinj = np.clip(
                    np.clip(pi_act[:, 0].flatten(), -1, 1) * bound_radius + midpoint,
                    state[:, 4] - PINJ_ACT_SCALE,
                    state[:, 4] + PINJ_ACT_SCALE,
                )
                bound_radius = (self._tinj_bounds[1] - self._tinj_bounds[0]) / 2
                midpoint = bound_radius + self._tinj_bounds[0]
                next_tinj = np.clip(
                    np.clip(pi_act[:, 1].flatten(), -1, 1) * bound_radius + midpoint,
                    state[:, 5] - TINJ_ACT_SCALE,
                    state[:, 5] + TINJ_ACT_SCALE,
                )
            next_pinj, next_tinj = self._beam_bounder.bound_beams(next_pinj, next_tinj)
            next_dw = self._momentum * state[:, 1] + (1 - self._momentum) * np.clip(
                next_pinj
                - TRANSITION_COEF * state[:, 0]
                * task[:, 2] ** -0.93
                * task[:, 1] ** -0.15
                * state[:, 4] ** 0.69,
                self._dw_bounds[0],
                self._dw_bounds[1]
            )
            next_w = np.clip(
                state[:, 0] + self._dt * state[:, 1],
                self._w_bounds[0],
                self._w_bounds[1]
            )
            next_drot = np.clip(
                next_tinj * task[:, 4]
                - state[:, 2] / task[:, 3],
                self._drot_bounds[0],
                self._drot_bounds[1]
            )
            next_rot = np.clip(
                state[:, 2] + self._dt * state[:, 3],
                self._rot_bounds[0],
                self._rot_bounds[1],
            )
            states[:, h + 1] = np.concatenate([
                next_w.reshape(-1, 1),
                next_dw.reshape(-1, 1),
                next_rot.reshape(-1, 1),
                next_drot.reshape(-1, 1),
                next_pinj.reshape(-1, 1),
                next_tinj.reshape(-1, 1),
            ], axis=1)
            obs[:, h + 1], rewards[:, h] = self._form_observation_and_rew(
                states[:, h + 1],
                targets,
                task,
                states[:, h]
            )
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
            'actuators': states[..., 4:],
        }

    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, max horizon).
        """
        return np.array([
            np.array([
                random.uniform(*self._betan_target_bounds),
                random.uniform(*self._rot_target_bounds)
            ])
            for _ in range(num_target_trajs)])

    def _form_observation(self, state, target, task=None, last_state=None):
        """Form the observation."""
        return self._form_observation_and_rew(state, target, task, last_state)[0]

    def _form_observation_and_rew(self, state, target, task=None, last_state=None):
        """Form the observation and the reward."""
        if task is None:
            task = self.get_task().reshape(1, -1)
        # Calculate the betan.
        to_convert = [state if len(state.shape) > 1 else state.reshape(1, -1)]
        if last_state is not None:
            to_convert.append(last_state if len(last_state.shape) > 1
                              else last_state.reshape(1, -1))
        betans = [
            tc[:, 0] * task[:, 0] * task[:, 1] / task[:, 2] * BETA_OBS_COEF
            for tc in to_convert
        ]
        # Calculate the difference in the target.
        bn_errs = betans[0] - target[:, 0]
        if self._bn_error_accum is None:
            self._bn_error_accum = bn_errs.reshape(1, -1)
        else:
            self._bn_error_accum = np.concatenate([
                self._bn_error_accum,
                bn_errs.reshape(1, -1)
            ], axis=0)
        rot_errs = state[:, 2] - target[:, 1]
        if self._rot_error_accum is None:
            self._rot_error_accum = rot_errs.reshape(1, -1)
        else:
            self._rot_error_accum = np.concatenate([
                self._rot_error_accum,
                rot_errs.reshape(1, -1)
            ], axis=0)
        # Calculate the I and D term.
        bn_dterm = np.zeros(len(task)) if last_state is None else betans[0] - betans[1]
        bn_dterm /= self._dt
        rot_dterm = (np.zeros(len(task)) if last_state is None
                     else state[:, 2] - last_state[:, 2])
        rot_dterm /= self._dt
        bn_iterm = np.sum(self._bn_error_accum, axis=0) * self._dt
        rot_iterm = np.sum(self._rot_error_accum, axis=0) * self._dt
        # Normalize terms in the observation and return.
        if self._obs_is_pid_only:
            obs = np.concatenate([
                -bn_errs.reshape(-1, 1) / BETAN_SIG,
                -rot_errs.reshape(-1, 1) / ROTATION_SIG,
                -bn_iterm.reshape(-1, 1) / BETAN_SIG,
                -rot_iterm.reshape(-1, 1) / ROTATION_SIG,
                -bn_dterm.reshape(-1, 1) / BETAN_SIG,
                -rot_dterm.reshape(-1, 1) / ROTATION_SIG,
            ], axis=1)
        elif self._include_pid_in_obs:
            obs = np.concatenate([
                (betans[0].reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
                (state[:, [2]] - ROTATION_MU) / ROTATION_SIG,
                -bn_errs.reshape(-1, 1) / BETAN_SIG,
                -rot_errs.reshape(-1, 1) / ROTATION_SIG,
                -bn_iterm.reshape(-1, 1) / BETAN_SIG,
                -rot_iterm.reshape(-1, 1) / ROTATION_SIG,
                -bn_dterm.reshape(-1, 1) / BETAN_SIG,
                -rot_dterm.reshape(-1, 1) / ROTATION_SIG,
                (target[:, [0]] - BETAN_MU) / BETAN_SIG,
                (target[:, [1]] - ROTATION_MU) / ROTATION_SIG,
                (state[:, [4]] - PINJ_MU) / PINJ_SIG,
                (state[:, [5]] - TINJ_MU) / TINJ_SIG,
            ], axis=1)
        else:
            obs = np.concatenate([
                (betans[0].reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
                (state[:, [2]] - ROTATION_MU) / ROTATION_SIG,
                (target[:, [0]] - BETAN_MU) / BETAN_SIG,
                (target[:, [1]] - ROTATION_MU) / ROTATION_SIG,
                (state[:, [4]] - PINJ_MU) / PINJ_SIG,
                (state[:, [5]] - TINJ_MU) / TINJ_SIG,
            ], axis=1)
        return obs, -np.abs(bn_errs) / BETAN_SIG - np.abs(rot_errs) / ROTATION_SIG

    def _beam_bounding(self, next_pinj, next_tinj):
        return next_pinj, next_tinj

    def plot_paths(self, rollouts, colors=('blue', 'red', 'green', 'orange', 'purple'),
                   is_pid=False):
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 2)
        for cidx, obs in enumerate(rollouts['observations']):
            if is_pid:
                axs[0, 0].plot(-obs[:, 0] * BETAN_SIG + rollouts['targets'][cidx, 0],
                               color=colors[cidx],
                               alpha=0.8, label=np.sum(rollouts['rewards'][cidx]))
                axs[0, 0].axhline(rollouts['targets'][cidx, 0],
                                  color=colors[cidx],
                                  ls='--')
                axs[0, 1].plot(-obs[:, 1] * ROTATION_SIG
                               + rollouts['targets'][cidx, 1],
                               color=colors[cidx],
                               alpha=0.8)
                axs[0, 1].axhline(rollouts['targets'][cidx, 1],
                                  color=colors[cidx],
                                  ls='--')
                axs[1, 0].plot(rollouts['actuators'][cidx, :, 0],
                               alpha=0.8, color=colors[cidx])
                axs[1, 0].axhline((self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
                                  + self._pinj_bounds[0], ls=':', color='black')
                axs[1, 1].plot(rollouts['actuators'][cidx, :, 1],
                               alpha=0.8, color=colors[cidx])
                axs[1, 1].axhline((self._tinj_bounds[1] - self._tinj_bounds[0]) / 2
                                  + self._tinj_bounds[0], ls=':', color='black')
            else:
                # Plot the Betan.
                axs[0, 0].axhline(obs[0, -4] * BETAN_SIG + BETAN_MU,
                                  color=colors[cidx], ls='--', alpha=0.6)
                axs[0, 0].plot(obs[:, 0] * BETAN_SIG + BETAN_MU,
                               color=colors[cidx], alpha=0.8,
                               label=np.sum(rollouts['rewards'][cidx]))
                axs[0, 1].axhline(obs[0, -3] * ROTATION_SIG + ROTATION_MU,
                                  color=colors[cidx], ls='--', alpha=0.6)
                axs[0, 1].plot(obs[:, 1] * ROTATION_SIG + ROTATION_MU,
                               color=colors[cidx], alpha=0.8)
                # Plot the powers.
                axs[1, 0].plot(obs[:, -2] * PINJ_SIG + PINJ_MU,
                               color=colors[cidx], alpha=0.6)
                axs[1, 1].plot(obs[:, -1] * TINJ_SIG + TINJ_MU,
                               color=colors[cidx], alpha=0.6)
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()


class CounterCurrentBeamBounding(gym.Env):

    def __init__(self):
        self.lb_domains = np.array([1716, 3441, 5403, 7248, 8998, 10680,
                                    11491, 13160, 14066])
        self.ub_domains = np.array([1716, 2621, 4290, 5101, 6784, 8534,
                                    10378, 12341, 14066])
        self.lb_slopes = np.array([-0.00097756, -0.00063725, 0.00063725,
                                   0.00063725, 0.00063725, 0.00097756,
                                   0.00097756, 0.00097756])
        self.ub_slopes = np.array([0.00097756, 0.00097756, 0.00097756,
                                   0.00063725, 0.00063725, 0.00063725,
                                   -0.00063725, -0.00097756])
        self.lb_offsets = np.array([3.355, 2.184, -4.702, -4.702, -4.702,
                                    -8.337, -8.337, -8.337])
        self.ub_offsets = np.array([0.0, 0.0, 0.0, 1.7362, 1.7362,
                                    1.7362, 14.963, 19.163])
        self.lb_vertices = np.array([[3.44114384e+03, -8.43253901e-03],
                                     [5.40360093e+03, -1.25900110e+00],
                                     [7.24832567e+03, -8.34570465e-02],
                                     [8.99821912e+03,  1.03165612e+00],
                                     [1.06805129e+04,  2.10369165e+00],
                                     [1.14916391e+04,  2.89662001e+00],
                                     [1.31609674e+04,  4.52849659e+00],
                                     [1.40661001e+04,  5.41332248e+00]])
        self.ub_vertices = np.array([[2.62139162e+03, 2.56258014e+00],
                                     [4.29071994e+03, 4.19445671e+00],
                                     [5.10184609e+03, 4.98738507e+00],
                                     [6.78413989e+03, 6.05942061e+00],
                                     [8.53403334e+03, 7.17453377e+00],
                                     [1.03787581e+04, 8.35007782e+00],
                                     [1.23412152e+04, 7.09950926e+00],
                                     [1.40661001e+04, 5.41332248e+00]])

    def bound_beams(self, next_pinj, next_tinj):
        is_scalar = isinstance(next_pinj, float)
        if is_scalar:
            next_pinj, next_tinj = np.array([next_pinj]), np.array([next_tinj])
        next_pinj *= 1e-3
        for btype in ['lb', 'ub']:
            # Figure out which of the requests violate the bounds.
            sgmts = np.argmin(
                next_pinj.reshape(-1, 1)
                >= getattr(self, f'{btype}_domains').reshape(1, -1),
                axis=1) - 1
            slopes = np.array([getattr(self, f'{btype}_slopes')[sg]
                               for sg in sgmts])
            offsets = np.array([getattr(self, f'{btype}_offsets')[sg]
                                for sg in sgmts])
            tinj_bdry = next_pinj * slopes + offsets
            if btype == 'lb':
                violations = tinj_bdry > next_tinj
            else:
                violations = tinj_bdry < next_tinj
            if np.sum(violations):
                # Project onto each of the boundaries of the polygon.
                offsets = getattr(self, f'{btype}_offsets')
                slopes = getattr(self, f'{btype}_slopes')
                domains = getattr(self, f'{btype}_domains')
                ortho_offsets = (next_tinj[violations].reshape(-1, 1)
                                 + 1 / slopes.reshape(1, -1)
                                 * next_pinj[violations].reshape(-1, 1))
                pinj_intersects = ((ortho_offsets - offsets.reshape(1, -1))
                                   / (slopes.reshape(1, -1) + 1
                                       / slopes.reshape(1, -1)))
                tinj_intersects = pinj_intersects * slopes + offsets
                # Figure out which of the boundaries we should project onto.
                invalid_projs = np.logical_or(
                    pinj_intersects < domains[:-1],
                    pinj_intersects > domains[1:]
                )
                pinj_intersects[invalid_projs] = np.inf
                vertices = getattr(self, f'{btype}_vertices')
                pinj_intersects = np.hstack([
                    pinj_intersects,
                    vertices[:, 0].reshape(1, -1).repeat(len(pinj_intersects), axis=0)
                ])
                tinj_intersects = np.hstack([
                    tinj_intersects,
                    vertices[:, 1].reshape(1, -1).repeat(len(tinj_intersects), axis=0)
                ])
                proj_idxs = np.argmin(np.sqrt(
                    (pinj_intersects - next_pinj[violations].reshape(-1, 1)) ** 2
                    + (tinj_intersects - next_tinj[violations].reshape(-1, 1)) ** 2
                ), axis=1)
                idxs = np.arange(len(proj_idxs))
                next_pinj[violations] = pinj_intersects[idxs, proj_idxs]
                next_tinj[violations] = tinj_intersects[idxs, proj_idxs]
        next_pinj *= 1e3
        if is_scalar:
            return float(next_pinj), float(next_tinj)
        return next_pinj, next_tinj
