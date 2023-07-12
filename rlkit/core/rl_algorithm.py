import abc
from collections import OrderedDict
import os

import gtimer as gt
import numpy as np
from omegaconf import DictConfig
import torch

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,  # Optionally sequence of them.
            replay_buffer: ReplayBuffer,
            early_stop_wait_epochs=None,
            early_stop_delta=None,
            early_stop_using_eval=True,
            use_gtimer=False,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        if isinstance(evaluation_env, DictConfig):
            evaluation_env = dict(evaluation_env)
        if not isinstance(evaluation_env, dict):
            evaluation_env = {'evaluation': evaluation_env}
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        if isinstance(evaluation_data_collector, DictConfig):
            evaluation_data_collector = dict(evaluation_data_collector)
        if not isinstance(evaluation_data_collector, dict):
            evaluation_data_collector = {'evaluation': evaluation_data_collector}
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        # Keep track of best expl and eval epochs.
        self._best_expl_epoch, self._best_eval_epoch = 0, 0
        self._best_expl_return, self._best_eval_return = float('-inf'), float('-inf')
        # Early stopping data.
        self._early_stop_wait_epochs = early_stop_wait_epochs
        self._early_stop_delta = early_stop_delta
        self._early_stop_using_eval = early_stop_using_eval
        self._early_stop_best_epoch = 0
        self._early_stop_best_return = float('-inf')
        self._should_early_stop = False
        # Gtimer stuff.
        self._use_gtimer = use_gtimer

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()
        logger.close()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _mark_returns(self, epoch, paths, eval_returns=False):
        score = np.mean([np.sum(path['rewards']) for path in paths])
        if eval_returns:
            if score > self._best_eval_return:
                self._best_eval_return = score
                self._best_eval_epoch = epoch
            if not self._early_stop_using_eval:
                return
        else:
            if score > self._best_expl_return:
                self._best_expl_return = score
                self._best_expl_epoch = epoch
            if self._early_stop_using_eval:
                return
        if (self._early_stop_wait_epochs is not None
                and self._early_stop_delta is not None):
            if score - self._early_stop_best_return > self._early_stop_delta:
                self._early_stop_best_return = score
                self._early_stop_best_epoch = epoch
            elif epoch - self._early_stop_best_epoch >= self._early_stop_wait_epochs:
                self._should_early_stop = True

    def _time_stamp(self, name, **kwargs):
        if self._use_gtimer:
            gt.stamp(name, **kwargs)

    def _end_epoch(self, epoch):
        for netname, net in self.trainer.get_snapshot().items():
            self._save_network_weights(netname + '.pt', net)
        if epoch == self._best_eval_epoch:
            for netname, net in self.trainer.get_snapshot().items():
                self._save_network_weights('best_eval_' + netname + '.pt', net)
        if epoch == self._best_expl_epoch:
            for netname, net in self.trainer.get_snapshot().items():
                self._save_network_weights('best_expl_' + netname + '.pt', net)
        # I found the below does not really work so I am just removing it in place of
        # my own function that just saves the network weights.
        # logger.save_itr_params(epoch, self._get_snapshot())
        logger.save_itr_networks(epoch, self.trainer.get_snapshot())
        self._time_stamp('saving', unique=False)
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        for edc in self.eval_data_collector.values():
            edc.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _save_network_weights(self, filename, net):
        save_path = os.path.join(logger.get_snapshot_dir(), filename)
        torch.save(self.expl_data_collector.policy.state_dict(), save_path)
        torch.save(net.state_dict(), save_path)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for edname, edc in self.eval_data_collector.items():
            for k, v in edc.get_snapshot().items():
                snapshot[f'{edname}/{k}'] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        for edname, edc in self.eval_data_collector.items():
            logger.record_dict(
                edc.get_diagnostics(),
                prefix=f'{edname}/',
            )
            eval_paths = edc.get_epoch_paths()
            eenv = self.eval_env[edname]
            if hasattr(eenv, 'get_diagnostics'):
                logger.record_dict(
                    eenv.get_diagnostics(eval_paths),
                    prefix=f'{edname}/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix=f"{edname}/",
            )

        """
        Misc
        """
        self._time_stamp('logging', unique=False)
        if self._use_gtimer:
            logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
