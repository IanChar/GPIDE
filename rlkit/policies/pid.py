"""
PID policy.

Author: Ian Char
Date: August 5, 2022
"""
from typing import Sequence, Tuple

import numpy as np
from rlkit.policies.base import Policy


class PIDController(Policy):
    """PID Controller."""

    def __init__(
        self,
        p_gain: float,
        i_gain: float,
        d_gain: float,
        baseline: float = 0,
    ):
        """Constructor."""
        self.gains = np.array([p_gain, i_gain, d_gain])
        self.baseline = baseline

    def get_action(self, observation):
        """Get the action."""
        return np.array([np.sum(self.gains * observation) + self.baseline]), np.ones(1)

    def get_actions(self, observations: np.ndarray) -> Tuple[np.ndarray]:
        """Get multiple actions.

        Args:
            observations: Shape (num_observations, obs dim).

        Returns: nparray of shape (num_observations, 1) and ones.
        """
        acts = np.sum(self.gains.reshape(1, -1) * observations, axis=1) + self.baseline
        return acts.reshape(-1, 1), np.ones(len(observations))


class MultiPIDController(Policy):
    """Collection of possibly several PD controllers. It is expected that whatever
    controller this is deployed on gives observations that look like
    [p1, p2, .., pD, i1, i2, ..., iD, d1, d2, ..., dD]
    """

    def __init__(
        self,
        p_gains: Sequence[float],
        i_gains: Sequence[float],
        d_gains: Sequence[float],
        baselines: Sequence[float] = (0.0, 0.0),
    ):
        """Constructor."""
        self.dims = len(p_gains)
        assert len(i_gains) == self.dims
        assert len(d_gains) == self.dims
        assert len(baselines) == self.dims
        self.gains = np.array([
            np.array(p_gains),
            np.array(i_gains),
            np.array(d_gains),
        ])
        self.baselines = np.array(baselines)

    def get_action(self, observation):
        assert len(observation) == 3 * self.dims
        observation = observation.reshape(3, self.dims)
        actions = np.sum(self.gains * observation, axis=0) + self.baselines
        return actions, np.ones(1)

    def get_actions(self, observations: np.ndarray) -> Tuple[np.ndarray]:
        """Get multiple actions.

        Args:
            observations: Shape (num_observations, obs dim).

        Returns: nparray of shape (num_observations, 1) and ones.
        """
        if len(observations.shape) == 1:
            observations = observations[np.newaxis]
        assert observations.shape[1] == 3 * self.dims
        observations = observations.reshape(observations.shape[0], 3, self.dims)
        acts = np.sum(self.gains * observations, axis=1) + self.baselines
        return acts, np.ones(len(observations))
