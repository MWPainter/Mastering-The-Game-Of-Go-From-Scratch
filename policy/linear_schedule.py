import numpy as np
from random import random


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = float(eps_begin)
        self.eps_begin      = float(eps_begin)
        self.eps_end        = float(eps_end)
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon
        for t=0, self.epsilon = self.eps_begin
        for t>=self.nsteps, self.epsilon = self.eps_end
        otherwise, linear decay

        Args:
            t: (int) nth frames
        """
        if t == 0: self.epsilon = self.eps_begin
        elif t >= self.nsteps: self.epsilon = self.eps_end
        else: 
            alpha = float(t) / float(self.nsteps)
            self.epsilon = (1.0 - alpha) * self.eps_begin + alpha * self.eps_end



class LinearExploration(LinearSchedule):
    def get_action(self, best_action, valid_actions):
        """
        Returns a randomi (valid) action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
            valid_actions: list of valid actions that we may take
        Returns:
            an action (after epsilon greedy)
        """
        if random() < self.epsilon:
            return np.random.choice(valid_actions, 1)[0]
        return best_action
