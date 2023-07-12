from collections import deque, OrderedDict

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.envs.env_utils import get_eval_settings_for_env


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            snapshot_env=True,
            **kwargs
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._snapshot_env = snapshot_env

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        if self._snapshot_env:
            return dict(
                env=self._env,
                policy=self._policy,
            )
        else:
            return dict(
                policy=self._policy,
            )

    @property
    def policy(self):
        """Get the policy that is used to collect data."""
        return self._policy


class EnvModelPathCollector(PathCollector):
    def __init__(
            self,
            model_env,
            policy,
            start_state_selector,
            max_num_epoch_paths_saved=None,
    ):
        self._model_env = model_env
        self._policy = policy
        self._start_state_selector = start_state_selector
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_starts = int(np.ceil(num_steps / max_path_length))
        starts = self._start_state_selector.get_starts(num_starts)
        obs, actions, rewards, terminals, envinfos, piinfos = self._model_env.unroll(
                starts, self._policy, max_path_length,
        )
        for pathnum in range(obs.shape[1]):
            pathlen = (len(terminals) if np.sum(terminals[:, pathnum]) == 0
                       else np.argmax(terminals[:, pathnum]))
            env_path_info, pi_path_info = \
                [[{} for _ in range(pathlen)] for _ in range(2)]
            for k, v in envinfos.items():
                for dtoadd, val in zip(env_path_info, v[:, pathnum]):
                    dtoadd[k] = val
            for k, v in piinfos.items():
                for dtoadd, val in zip(pi_path_info, v[:, pathnum]):
                    dtoadd[k] = val
            paths.append(dict(
                    observations=obs[:pathlen, pathnum],
                    next_observations=obs[1:pathlen + 1, pathnum],
                    actions=actions[:pathlen, pathnum],
                    rewards=rewards[:pathlen, pathnum].reshape(-1, 1),
                    terminals=terminals[:pathlen, pathnum].reshape(-1, 1),
                    env_infos=env_path_info,
                    agent_infos=pi_path_info,
            ))
        self._num_paths_total += len(paths)
        self._num_steps_total += num_starts * max_path_length
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            policy=self._policy,
        )

    @property
    def policy(self):
        """Get the policy that is used to collect data."""
        return self._policy


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

    @property
    def policy(self):
        """Get the policy that is used to collect data."""
        return self._policy


class BatchMdpPathCollector(MdpPathCollector):

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        num_paths = int(np.ceil(num_steps // max_path_length))
        unrolls = self._env.rollout(
            policy=self._policy,
            num_rollouts=num_paths,
            horizon=max_path_length,
        )
        # Convert dict with everything to list of dicts
        paths = []
        for pidx in range(unrolls['observations'].shape[0]):
            pathlen = (max_path_length if np.sum(unrolls['terminals'][pidx]) == 0
                       else np.argmax(unrolls['terminals'][pidx]) + 1)
            env_path_info, pi_path_info = \
                [[{} for _ in range(pathlen)] for _ in range(2)]
            for log_pi, pi_info in zip(unrolls['logpi'][pidx], pi_path_info):
                pi_info['logpi'] = log_pi
            path_to_add = {}
            for k in ['observations', 'next_observations', 'actions', 'rewards',
                      'terminals']:
                path_to_add[k] = unrolls[k][pidx, :pathlen]
            path_to_add['env_infos'] = env_path_info
            path_to_add['agent_infos'] = pi_path_info
            paths.append(path_to_add)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_paths * max_path_length
        self._epoch_paths.extend(paths)
        return paths


class TaskEvalCollector(MdpPathCollector):

    def __init__(
        self,
        env,
        policy,
        env_name,
        split,
        settings=None,
        **kwargs,
    ):
        self.env_name = env_name
        if settings is None:
            settings = get_eval_settings_for_env(env_name, split=split)
        else:
            settings = settings
        self.num_settings = len(list(settings.values())[0])
        if 'fusion' in env_name:
            self.settings = [{k: settings[k][[idx]] for k in settings.keys()}
                             for idx in range(len(settings['start']))]
        else:
            self.settings = [{k: settings[k][idx] for k in settings.keys()}
                             for idx in range(len(settings['start']))]
        super().__init__(env=env, policy=policy, **kwargs)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        """Note that num_steps will essentially be ignored since we will do
        the max path length for every setting."""
        paths = []
        num_steps_collected = 0
        for setting in self.settings:
            max_path_length_this_loop = max_path_length
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                reset_kwargs=setting,
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths


class BatchTaskEvalCollector(MdpPathCollector):

    def __init__(
        self,
        env,
        policy,
        env_name,
        split,
        settings=None,
        **kwargs,
    ):
        """
        Collect rollouts for a set number of tasks.

        Args:
            env: The gym environment.
            policy: The from the Policy class.
            env_name: Name of the environment.
            validation_tasks: Whether we are using evaluation tasks.
            settings: If the settings are already loaded in use these. Otherwise
                this will load them in.
        """
        if settings is None:
            self.settings = get_eval_settings_for_env(env_name, split=split)
        else:
            self.settings = settings
        self.num_settings = len(list(self.settings.values())[0])
        super().__init__(env=env, policy=policy, **kwargs)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        unrolls = self._env.rollout(
            policy=self._policy,
            num_rollouts=self.num_settings,
            horizon=max_path_length,
            **self.settings
        )
        num_paths = len(unrolls['observations'])
        # Convert dict with everything to list of dicts
        paths = []
        for pidx in range(unrolls['observations'].shape[0]):
            pathlen = (max_path_length if np.sum(unrolls['terminals'][pidx]) == 0
                       else np.argmax(unrolls['terminals'][pidx]) + 1)
            env_path_info, pi_path_info = \
                [[{} for _ in range(pathlen)] for _ in range(2)]
            for log_pi, pi_info in zip(unrolls['logpi'][pidx], pi_path_info):
                pi_info['logpi'] = log_pi
            path_to_add = {}
            for k in ['observations', 'next_observations', 'actions', 'rewards',
                      'terminals']:
                path_to_add[k] = unrolls[k][pidx, :pathlen]
            path_to_add['env_infos'] = env_path_info
            path_to_add['agent_infos'] = pi_path_info
            paths.append(path_to_add)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_paths * max_path_length
        self._epoch_paths.extend(paths)
        return paths
