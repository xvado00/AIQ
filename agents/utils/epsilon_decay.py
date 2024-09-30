class EpsilonDecayMixin:
    """
    Utility mixin for epsilon decay in epsilon greedy agents.
    Epsilon = probability of choosing random action
    Epsilon is in <0, 1>
    """

    def __init__(self, min_epsilon: float, episodes_till_min_decay: float):
        """
        Epsilon allways starts at 1.0
        :param min_epsilon: set min_epsilon = 1.0 to turn off the decay function
        :param episodes_till_min_decay: how many episodes should it take to decay to min_epsilon
        """
        assert 0 <= min_epsilon <= 1
        assert 0 <= episodes_till_min_decay

        self.min_epsilon = min_epsilon
        self.episodes_till_min_decay = int(episodes_till_min_decay)

        self.epsilon = 1.0

        self._linear_decay_rate = None
        if self.episodes_till_min_decay != 0:
            self._linear_decay_rate = (self.epsilon - self.min_epsilon) / self.episodes_till_min_decay
        # Has to be like this
        # self.reset() -> resets whole base class not just the mixin
        EpsilonDecayMixin.reset(self)

    def decay_epsilon_linear(self):
        """
        Decrements epsilon by calculated step until it hits self.min_epsilon
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self._linear_decay_rate
        else:
            self.epsilon = self.min_epsilon

    def reset(self):
        self.epsilon = 1.0
        if self.episodes_till_min_decay == 0:
            self.epsilon = self.min_epsilon
