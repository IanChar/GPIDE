"""
Sequence replay buffer implemented in a naive but easy way.

Author: Ian Char
Date: January 20, 2022
"""
from collections import OrderedDict
from typing import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class NaiveSequenceReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        batch_window_size: int,
    ):
        """
        Constructor.

        Args:
            max_replay_buffer_size: The maximum size of the replay buffer. One data
                point refers to one sequence of the data.
            env: The environment that experience is being collected from.
            batch_window_size: The size of window that are in a batch.
        """
        self._observation_dim = get_dim(env.observation_space)
        self._action_dim = get_dim(env.action_space)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._window_size = batch_window_size
        self.clear_buffer()

    def clear_buffer(self):
        """Clear all of the buffers."""
        self._observations = np.zeros((
            self._max_replay_buffer_size,
            self._window_size,
            self._observation_dim,
        ))
        self._next_observations = np.zeros((
            self._max_replay_buffer_size,
            self._window_size,
            self._observation_dim,
        ))
        # NOTE: actions and rewards have one padding at the beginning. This is because
        # when encoding for the first time step we need to encode previous actions
        # and rewards but there are none at that point.
        self._actions = np.zeros((
            self._max_replay_buffer_size,
            self._window_size + 1,
            self._action_dim,
        ))
        self._rewards = np.zeros((
            self._max_replay_buffer_size,
            self._window_size + 1,
            1,
        ))
        self._terminals = np.zeros((
            self._max_replay_buffer_size,
            self._window_size,
            1,
        ), dtype='uint8')
        self._masks = np.zeros((
            self._max_replay_buffer_size,
            self._window_size,
            1,
        ), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_path(self, path: Dict[str, np.ndarray]):
        """
        Add a path to the replay buffer.

        Args:
            path: The path collected as a dictionary of ndarrays.
        """
        length = len(path['actions'])
        for strt in range(0, max(length - self._window_size + 1, 1)):
            end = min(strt + self._window_size, strt + length)
            for k, buff in (('actions', self._actions), ('rewards', self._rewards)):
                buff[self._top, 1:end - strt + 1] = path[k][strt:end]
            for k, buff in (
                    ('observations', self._observations),
                    ('next_observations', self._next_observations),
                    ('terminals', self._terminals)):
                buff[self._top, :end - strt] = path[k][strt:end]
            self._masks[self._top, :end - strt] = 1
            self._masks[self._top, end - strt:] = 0
            self._advance()

    def random_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Dictionary of information for the update.
            observations: This is a history w shape (batch_size, L, obs_dim)
            actions: This is the history of actions (batch_size, L + 1, act_dim)
            rewards: This is the rewards at the last point (batch_size, L + 1, 1)
            next_observation: This is a history of nexts (batch_size, L, obs_dim)
            terminals: Whether last time step is terminals (batch_size, L, 1)
            masks: Masks of what is real and what data (batch_size, L, 1).
        """
        indices = np.random.randint(0, self._size, batch_size)
        batch = {}
        batch['observations'] = self._observations[indices]
        batch['next_observations'] = self._next_observations[indices]
        batch['actions'] = self._actions[indices]
        batch['rewards'] = self._rewards[indices]
        batch['terminals'] = self._terminals[indices]
        batch['masks'] = self._masks[indices]
        return batch

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        raise NotImplementedError('This buffer does not support adding single samples')

    def num_steps_can_sample(self):
        return self._size

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
