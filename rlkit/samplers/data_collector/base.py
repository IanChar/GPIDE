import abc


class DataCollector(object, metaclass=abc.ABCMeta):
    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    @property
    @abc.abstractmethod
    def policy(self):
        """Get the policy that is used to collect data."""

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass


class PathCollector(DataCollector, metaclass=abc.ABCMeta):

    def eval(self, eval_mode):
        if hasattr(self, '_policy'):
            if hasattr(self._policy, 'deterministic'):
                self._policy.deterministic = eval_mode

    @abc.abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass


class StepCollector(DataCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass
