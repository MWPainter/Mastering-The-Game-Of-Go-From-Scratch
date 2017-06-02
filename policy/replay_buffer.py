import numpy as np
import random
from collections import deque 

def prod(l):
    """
    Helper to multiply all of the elements in a list
    """
    return reduce(lambda x,y: x*y, l) 


class ReplayBufferError(Exception):
    """
    Exception to throw if we incorrectly use the replay buffer!
    """
    pass


class ReplayBuffer(object):
    """
    Replay buffer used to sample mini batches of SARS examples for 
    """
    def __init__(self, size, s_shape, a_shape, r_shape):
        """
        Args:
            size: the size of the replay buffer
            s_shape: shape of states to be used
            a_Shape: shape of actions to be used
            r_shape: shape of rewards to be used
        """
        self._size = size
        self._contains = 0
        self._s_shape = s_shape
        self._s_size = prod(s_shape)
        self._a_shape = a_shape
        self._a_size = prod(a_shape)
        self._r_shape = r_shape
        self._r_size = prod(r_shape)
        self._sars_size = self._s_size * 2 + self._a_size * self._r_size
        self._queue = np.zeros((self.size, self.sars_size))


    @property
    def _is_full(self):
        """
        self._contains is kept updated with how many examples we've seen until the replay buffer actually becomes full
        we shouldn't sample from the replay buffer until the replay buffer has actually been filled
        """
        return self._size <= self._contains


    def _pop(self, n):
        """
        Pops n sars samples from the queue. After this the last n sars samples in self.queue will be garbage memory
        Args:
            n: the number of sars samples to remove
        """
        self._queue[:-n] = self._queue[n:]


    def _push(self, sarses):
        """
        Pushes sarses onto the end of the queue
        This should immediately follow a pop call
        Args:
            sarses: A numpy ndarray of shape (-1,_sars_size) to be pushed onto the end of the queue
        """
        sarses_len = sarses.shape[0]
        self._queue[-sarses_len:] = sarses
        

    def _encode_sars(self, s, a, r, sp):
        """
        Encodes numpy ndarrays, s, a, r, sp, into a single sarses array of the same length
        Assumes s, a, r, sp are arrays of the same length
        Args:
            s: np.ndarray of states (sars examples)
            a: np.ndarray of actions (sars examples)
            r: np.ndarray of rewards (sars examples)
            sp: np.ndarray of states (sars examples)
        Return:
            sarses: np.ndarray of sars examples (flattened/encoded)
        """
        # TODO: flatten each, then concatenate


    def _decode_sars(self, sarses):
        """
        Inverse of the _encode_sars function
        Args:
            sarses: np.ndarray of sars examples (flattened/encoded)
        Returns:
            tuple: (s,a,r,sp), each an np.ndarray of s/a/r/sp parts of a SARS example
        """
        # TODO: chop up, and reshape
        

    def store_example(self, s, a, r, sp):
        """
        Adds one sars example to the internal queue. Has the side effect of removing the oldest example from the buffer 
        if the buffer is full
        Args:
            s: initial state of the sars sample
            a: action of the sample
            r: reward of the sample
            sp: next state of the sars sample
        """
        # TODO: Extend first dim of s, a, r, sp and call batch one
        

    def store_example_batch(self, s_arr, a_arr, r_arr, sp_arr):
        """
        Same as store_example, except each of s, a, r, sp is an array 
        Args:
            s_arr: np.ndarray of initial state of the sars samples
            a_arr: np.ndarray of actions of the samples
            r_arr: np.ndarray of reward of the samples
            sp_arr: np.ndarray of next state of the sars samples
        """
        sarses = self._encode_sars(s_arr, a_arr, r_arr, sp_arr)
        sarses_len = sarses.shape[0]
        if not self.is_full: 
            self._contains += sarses_len
        self._pop(sarses_len)
        self._push(sarses)

    def sample(self, n):
        """
        Sample n examples from the replay buffer
        Throws an exception if the replay buffer isn't properly initialized (i.e. the internal queue isn't full)
        Args:
            n: the number of examples to sample
        Returns:
            tuple: (s's, a's, r's, sp's), each being a numpy ndarray of length n in the first dimension
        """
        if not self.is_full or n >= self.size:
            raise ReplayBufferError
        indices = np.random.choice(self.size, n, replace=False)
        sarses_samples = self._queue[indices]
        return self._decode_sarses(sarses_samples)


