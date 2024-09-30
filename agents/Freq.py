
#
# Simple frequence based agent for testing in AIQ
#
# Copyright Shane Legg 2011
# Copyright Petr Zeman 2023
# Copyright Jan Štipl 2024
# Released under GNU GPLv3
#

from .Agent import Agent

from numpy import zeros
from numpy import ones

import  numpy as np


from random import randint, randrange, random

from .utils.epsilon_decay import EpsilonDecayMixin


class Freq(Agent, EpsilonDecayMixin):

    def __init__( self, refm, disc_rate, min_epsilon=0.05, episodes_till_min_decay=0 ):
        """
        :param refm:
        :param disc_rate:
        :param episodes_till_min_decay: Steps over which epsilon decays to min_epsilon; 0 turns off decay
        :param min_epsilon: Minimum value of epsilon for exploration-exploitation trade-off
        """

        Agent.__init__( self, refm, disc_rate )
        EpsilonDecayMixin.__init__( self, min_epsilon=min_epsilon, episodes_till_min_decay=episodes_till_min_decay )
        
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells   = refm.getNumObsCells()

        self.reset()


    def reset( self ):
        EpsilonDecayMixin.reset(self)
        self.action = 0

        self.total = zeros( (self.num_actions) )
        self.acts  = ones(  (self.num_actions) )


    def __str__( self ):
        return (f"Freq({self.min_epsilon},"
                f"{self.episodes_till_min_decay})")


    def perceive( self, observations, reward ):

        if len(observations) != self.obs_cells:
            raise NameError("Freq received wrong number of observations!")

        # set up alisas
        Total = self.total
        Acts  = self.acts

        Total[self.action] += reward
        Acts[self.action] += 1
        
        # find an optimal action according to mean reward for each action
        opt_action = self.random_optimal( Total/Acts )

        # action selection
        if self.sel_mode == 0:
            # do an epsilon greedy selection
            if random() < self.epsilon:
                naction = randrange(self.num_actions)
            else:
                naction = opt_action
        else:
            # do a softmax selection
            naction = self.soft_max( Total/Acts, self.epsilon )

 
        # update the old action
        self.action = naction
        self.decay_epsilon_linear()

        return naction

