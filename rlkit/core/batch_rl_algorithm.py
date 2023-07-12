import abc

import gtimer as gt
import numpy as np
from tqdm import tqdm

from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            clear_buffer_every_train_loop=False,
            early_stop_wait_epochs=None,
            early_stop_delta=None,
            early_stop_using_eval=True,
            use_gtimer=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            early_stop_wait_epochs=early_stop_wait_epochs,
            early_stop_delta=early_stop_delta,
            early_stop_using_eval=early_stop_using_eval,
            use_gtimer=use_gtimer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.clear_buffer_every_train_loop = clear_buffer_every_train_loop

    def _train(self):
        self.training_mode(False)
        self.trainer.training_mode(False)
        if self.min_num_steps_before_training > 0:
            self.expl_data_collector.eval(False)
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        epoch_iterator = range(self._start_epoch, self.num_epochs)
        if self._use_gtimer:
            epoch_iterator = gt.timed_for(
                epoch_iterator,
                save_itrs=True,
            )
        for epoch in epoch_iterator:
            for _ in tqdm(range(self.num_train_loops_per_epoch)):
                self.expl_data_collector.eval(False)
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                self._time_stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                self._time_stamp('data storing', unique=False)

                self.training_mode(True)
                self.trainer.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                self._time_stamp('training', unique=False)
                self.training_mode(False)
                self.trainer.training_mode(False)
                if self.clear_buffer_every_train_loop:
                    self.replay_buffer.clear_buffer()
            self._mark_returns(epoch, new_expl_paths, eval_returns=False)
            for edname, edc in self.eval_data_collector.items():
                edc.eval(True)
                eval_paths = edc.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
                if edname == 'evaluation':
                    self._mark_returns(epoch, eval_paths, eval_returns=True)
            self._time_stamp('evaluation sampling', unique=False)
            self._end_epoch(epoch)
            if self._should_early_stop:
                break
