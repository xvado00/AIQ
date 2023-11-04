
#
# Simple agent that allways performs specified action for testing in AIQ
#
# Copyright Jan Štipl 2023
# Released under GNU GPLv3
#

from typing import Generator

from refmachines.ReferenceMachine import ReferenceMachine
from .Agent import Agent


def always_same(constant_action: int) -> Generator[int, None, None]:
    """
    Generator that returns given constant action
    :param constant_action:
    :return:
    """
    while True:
        yield constant_action


class Constant(Agent):
    """
    Agent that returns given constant action regardless of state

    This agent is good for baseline measurements
    """

    def __init__(self, refm: ReferenceMachine, disc_rate, constant_action: float):
        Agent.__init__(self, refm, disc_rate)
        # Cast the parameter to int (other agents need floats)
        self.action = int(constant_action)

        # Action has to be valid
        assert 0 <= constant_action < self.num_actions, \
            f"Given action must be 0 <= {constant_action} < {self.num_actions} "

        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()

        self.generator: Generator[int, None, None] = always_same(self.action)

    def __str__(self):
        return "Constant(" + str(self.action) + ")"

    def reset(self):
        """
        There is nothing to reset
        :return:
        """
        pass

    def perceive(self, new_obs=None, reward=None) -> int:
        action: int = next(self.generator)

        return action
