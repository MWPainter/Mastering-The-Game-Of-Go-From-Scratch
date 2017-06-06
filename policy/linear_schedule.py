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

# Helper for get_action function
def _get_valid_action_indices(env):
    """
    Get's all of the open board positions.
    Converts those coordinates to action numbers
    Returns that lis

    Args:
        env: (open AI environment) 
    Returns:
        List of valid actions that an agent could take
    """
    # Get open board position
    state = env.state.board.encode()            
    board_size = state.shape[0]
    free_spaces = state[2]
    non_zero_coords = np.transpose(np.nonzero(free_spaces))

    # Get action numbers
    non_zero_coords[:,0] *= board_size
    actions = np.sum(non_zero_coords, axis=1)

    # Return
    return actions


class LinearExploration(LinearSchedule):
    def get_action(self, best_action, env):
        """
        Returns a randomi (valid) action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
            env: (open AI environment) used to sample a random (valid) action if necessary
        Returns:
            an action (after epsilon greedy)
        """
        if random() < self.epsilon:
            actions = _get_valid_action_indices(env)
            return np.random.choice(actions, 1)[0]
        return best_action
