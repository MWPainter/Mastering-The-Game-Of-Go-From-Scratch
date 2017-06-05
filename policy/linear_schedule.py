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
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action (after epsilon greedy)
        """
        if random() < self.epsilon:
            return self.env.action_space.sample()
        return best_action
