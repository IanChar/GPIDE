import abc


class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class SequencePolicy(object, metaclass=abc.ABCMeta):
    """
    General policy that takes in a sequence of past observations, actions, rewards
    isntead of just the current observation.
    """
    @abc.abstractmethod
    def get_action(self, observations, actions, rewards):
        """Get an action

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass


class SeqeunceExplorationPolicy(SequencePolicy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass
