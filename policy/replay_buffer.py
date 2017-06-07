import numpy as np
import functools

def prod(l):
    """
    Helper to multiply all of the elements in a tuple
    """
    return functools.reduce(lambda x,y: x*y, (1,) + l) 


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
        self._d_shape = () # done mask only has 0/1 values
        self._d_size = 1
        self._v_shape = () # values are floats
        self._v_size = 1
        self._sars_size = self._s_size * 2 + self._a_size \
                        + self._r_size + self._d_size + self._v_size
        self._queue = np.zeros((self._size, self._sars_size))


    @property
    def should_sample(self):
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
        

    def _encode_sarses(self, s, a, r, sp, d, v):
        """
        Encodes numpy ndarrays, s, a, r, sp, into a single sarses array of the same length
        Assumes s, a, r, sp are arrays of the same length
        Args:
            s: np.ndarray of states (sars examples)
            a: np.ndarray of actions (sars examples)
            r: np.ndarray of rewards (sars examples)
            sp: np.ndarray of states (sars examples)
            d: np.ndarray for done mask (if sars example is terminal)
            v: np.ndarray for (empirical) value
        Return:
            sarses: np.ndarray of sars examples (flattened/encoded)
        """
        batch_size = s.shape[0]
        if batch_size != a.shape[0] or \
           batch_size != r.shape[0] or \
           batch_size != sp.shape[0] or \
           batch_size != d.shape[0] or \
           batch_size != v.shape[0]:
            raise ReplayBufferError

        s = np.reshape(s, (batch_size, -1))
        a = np.reshape(a, (batch_size, -1))
        r = np.reshape(r, (batch_size, -1))
        sp = np.reshape(sp, (batch_size, -1))
        d = np.reshape(d, (batch_size, -1))
        v = np.reshape(v, (batch_size, -1))

        sarses = np.concatenate((s,a,r,d,v,sp), axis=-1)
        return sarses


    def _decode_sarses(self, sarses):
        """
        Inverse of the _encode_sars function
        Args:
            sarses: np.ndarray of sars examples (flattened/encoded)
        Returns:
            tuple: (s,a,r,sp,d,v), each an np.ndarray of s/a/r/sp parts of a SARS example 
                    and their corresponding done mask (d) and value (v)
        """
        s_div = self._s_size
        a_div = self._a_size + s_div
        r_div = self._r_size + a_div
        d_div = self._d_size + r_div
        v_div = self._v_size + d_div

        s =  sarses[:,      :s_div]
        a =  sarses[:, s_div:a_div]
        r =  sarses[:, a_div:r_div]
        d =  sarses[:, r_div:d_div]
        v =  sarses[:, d_div:v_div]
        sp = sarses[:, v_div:     ]

        s = np.reshape(s, (-1,) + self._s_shape)
        a = np.reshape(a, (-1,) + self._a_shape)
        r = np.reshape(r, (-1,) + self._r_shape)
        sp = np.reshape(sp, (-1,) + self._s_shape)
        d = np.reshape(d, (-1,) + self._d_shape)
        v = np.reshape(v, (-1,) + self._v_shape)

        return (s,a,r,sp,d,v)


        

    def store_example(self, s, a, r, sp, done, value):
        """
        Adds one sars example to the internal queue. Has the side effect of removing the oldest example from the buffer 
        if the buffer is full
        Args:
            s: initial state of the sars sample
            a: action of the sample
            r: reward of the sample
            sp: next state of the sars sample
            done_mask: 1.0 if done, 0.0 if not
            value: true value of state s (note that in real life gamma = 1.0, or more like 0.99)
        """
        s = np.expand_dims(s, axis=0)
        a = np.expand_dims(a, axis=0)
        r = np.expand_dims(r, axis=0)
        sp = np.expand_dims(sp, axis=0)
        d = np.expand_dims(done, axis=0)
        v = np.expand_dims(value, axis=0)
        self.store_example_batch(s,a,r,sp,d,v)
        

    def store_example_batch(self, s_arr, a_arr, r_arr, sp_arr, done_mask, v_arr):
        """
        Same as store_example, except each of s, a, r, sp is an array 
        Args:
            s_arr: np.ndarray of initial state of the sars samples
            a_arr: np.ndarray of actions of the samples
            r_arr: np.ndarray of reward of the samples
            sp_arr: np.ndarray of next state of the sars samples
            done_mask: np.ndarray for done mask. 1.0 if the sars example is a terminal one
            v_arr: np.ndarry of values for the state (we know the actual value of a game after its done)
        """
        s_arr = np.array(s_arr)
        a_arr = np.array(a_arr)
        r_arr = np.array(r_arr)
        sp_arr = np.array(sp_arr)
        done_mask = np.array(done_mask)
        v_arr = np.array(v_arr)
        sarses = self._encode_sarses(s_arr, a_arr, r_arr, sp_arr, done_mask, v_arr)
        sarses_len = sarses.shape[0]
        if not self.should_sample: 
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
        if not self.should_sample or n > self._size:
            raise ReplayBufferError
        indices = np.random.choice(self._size, n, replace=False)
        sarses_samples = self._queue[indices]
        return self._decode_sarses(sarses_samples)



"""
Include a main function for some testing
Manually check that the values make sense
"""
if __name__ == "__main__":
    rb = ReplayBuffer(3, (2,2), (1,), (1,))
    ss = []
    aa = []
    rr = []
    spsp = []
    dd = []
    vv = []
    for i in range(3):
        s = np.array([[i,i],[i,i]])
        a = np.array([i])
        r = np.array([i])
        sp = np.array([[i+1,i],[i,i+1]])
        d = np.array([0.0] if i != 2 else [1.0])
        v = np.array([-1.0] if i != 1 else [1.0])
        ss.append(s)
        aa.append(a)
        rr.append(r)
        spsp.append(sp)
        dd.append(d)
        vv.append(v)
    rb.store_example_batch(ss, aa, rr, spsp, dd, vv)
    print(rb._queue)
    print("Sample 1 (5 times):")
    for _ in range(5): print(rb.sample(1))
    print("\n")
    print("Sample 3:")
    print(rb.sample(3))
    print("\n")
    rb.store_example(np.array([[3,3,],[3,3]]),
                     np.array([3]),
                     np.array([3]),
                     np.array([[3,3],[3,4]]),
                     [1.0],
                     [0.0])
    rb.store_example(np.array([[3,3,],[3,3]]),
                     np.array([3]),
                     np.array([3]),
                     np.array([[3,3],[3,3]]),
                     [1.0],
                     [0.0])
    print("Sample 3 (with 2 changed):")
    print(rb.sample(3))
    print("\n"  )
    print("Intrnal queue (check is of length 3):")
    print(rb._queue)


