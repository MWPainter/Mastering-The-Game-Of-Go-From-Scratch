import gym


"""
Go object representing the state of a go board, to be used with the MCTS implementation
This is little more than a wrapper around the OpenAI Gym's implementation of Go.
"""
class Go(object):
    def __init__(self, boardsize=19):
        self.env
