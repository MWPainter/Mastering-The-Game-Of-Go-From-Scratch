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
    def __init__(self, size, s_shape, a_shape, r_shape, board_size):
        """
        Args:
            size: the size of the replay buffer
            s_shape: shape of states to be used
            a_Shape: shape of actions to be used
            r_shape: shape of rewards to be used
            board_size: the size of the board that we're a replay buffer for
        """
        self._size = size
        self._contains = 0
        self._board_size = board_size
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


    def _action_to_coord(self, action):
        """
        Helper to convert an action to board position
        
        Args:
            action: np.ndarray of actions, shape of (n,)
        Returns:
            coords: np.ndarray of coords, shape of (n,2)
        """
        n = action.shape[0]
        coords = np.zeros((n,2))
        coords[:,0] = np.mod(action, self._board_size)           # x's
        coords[:,1] = np.floor_divide(action, self._board_size)  # y's
        return coords


    def _coord_to_action(self, coord):
        """
        Helper to convert a board position to an action number

        Args:
            coords: np.ndarray of coords, shape of (n,2)
        Returns:
            action: np.ndarray of actions, shape of (n,)
        """
        return np.flatten(action[:,0] * self._board_size + action[:,1])


    def _reflect_coord(self, coord):
        """
        Returns reflected coords 

        Args:
            coords: np.ndarray of coords, shape of (n,2)
        Returns:
            coords: reflected coords
        """
        coords[:,1] = self._board_size - coords[:,1]
        return coords

    
    def _rotate_coord(self, coord, rot):
        """
        Returns a rotated coord. We rotate 'rot' many times 90 degrees clockwise, and assume origin is in top left
        The grid is of coords from 0, ..., _board_size - 1
        So the transform we want is
        (x,y) -> (board_size - 1 - y, x)

        Args:
            coords: np.ndarray of coords, shape of (n,2)
        Returns:
            coords: rotated coords
        """
        for _ in range(rot):
            next_coord = np.copy(coord)
            next_coord[:,0] = self._board_size - 1 - coord[:,1]
            next_coord[:,1] = coord[:,0]
            coord = next_coord
        return coord


    def _reflect_state(self, state):
        """
        Retrurns reflected states, (reverse in the last dimension)
        Note that if a = [0,1,2,3,4,5]
        then a[::2] prints out every 2nd element
        so a[::2] == [0,2,4]
           a[1::2] == [1,3,5]

        Args:
            s_arr: np.ndarray of initial state of the sars samples
                    shape = (n, 3, s, s)
        Returns:
            s_arr: reflected states
                    shape = (n, 3, s, s)
        """
        return state[:,:,:,::-1]
        
    
    def _apply_transform(self, reflect, rot, s_arr, a_arr, sp_arr):
        """
        Apply a transform of reflection/rotation

        Args:
            reflect: bool, if should reflect the board
            rot: the number of 90 degree rotations to make
            s_arr: np.ndarray of initial state of the sars samples
                    shape = (n, 3, s, s)
            a_arr: np.ndarray of actions of the samples
                    shape = (n,)
            sp_arr: np.ndarray of next state of the sars samples
                    shape = (n, 3, s, s)
        Returns:
            s_arr: np.ndarray of transformed states
            a_arr: np.ndarray of transformed actions
            sp_arr: np.ndarray of transformaed successor states
        """
        # Copys, so don't clobber old version
        s_arr_trans = np.copy(s_arr)
        a_arr_trans = np.copy(a_arr)
        sp_arr_trans = np.copy(sp_arr)

        # Convert to coord, so can do the math for actions
        coords = self._action_to_coord(a_arr_trans)

        # Apply reflections
        if reflect:
            s_arr_tran = self._reflect_state(s_arr_tran)
            coords = self._reflect_coord(coords)
            sp_arr_tran = self._reflect_state(sp_arr_tran)

        # Apply rotations
        s_arr_tran = np.rot90(s_arr_tran, rot, (3,2))
        coords = self._rotate_coord(coords, rot)
        sp_arr_tran = np.rot90(sp_arr_tran, rot, (3,2))

        # Conver back to actions from coords
        a_arr_trans = self._coord_to_action(coords)

        return s_arr_trans, a_arr_trans, sp_arr_trans


    def _exploit_symmetries(self, s_arr, a_arr, r_arr, sp_arr, d_arr, v_arr):
        """
        Takes a list of examples, an replicates them using rotational and reflectional symmetries
        them to provide more training examples

        Note that states and actions need to reflect the rotations/reflections. Rewards, the done mask 
        and value are invarient to the orientation of the board

        Args:
            s_arr: np.ndarray of initial state of the sars samples
                    shape = (n, 3, s, s)
            a_arr: np.ndarray of actions of the samples
                    shape = (n,)
            r_arr: np.ndarray of reward of the samples
                    shape = (n,)
            sp_arr: np.ndarray of next state of the sars samples
                    shape = (n, 3, s, s)
            done_mask: np.ndarray for done mask. 1.0 if the sars example is a terminal one
                    shape = (n,)
            v_arr: np.ndarry of values for the state (we know the actual value of a game after its done)
                    shape = (n,)
        Returns:
            s_arr: np.ndarray of initial state of the sars samples
                    shape = (8n, 3, s, s)
            a_arr: np.ndarray of actions of the samples
                    shape = (8n,)
            r_arr: np.ndarray of reward of the samples
                    shape = (8n,)
            sp_arr: np.ndarray of next state of the sars samples
                    shape = (8n, 3, s, s)
            done_mask: np.ndarray for done mask. 1.0 if the sars example is a terminal one
                    shape = (8n,)
            v_arr: np.ndarry of values for the state (we know the actual value of a game after its done)
                    shape = (8n,)
        """
        old_arr_len = s_arr.shape[0]
        arr_len = 8 * old_arr_len

        new_s_arr = np.zeros((arr_len,) + self._s_shape)
        new_a_arr = np.zeros((arr_len,))
        new_r_arr = np.zeros((arr_len,))
        new_sp_arr = np.zeros((arr_len,) + self._s_shape)
        new_d_arr = np.zeros((arr_len,))
        new_v_arr = np.zeros((arr_len,))

        for i in range(8):
            beg = old_arr_len * i
            end = old_arr_len * (i+1)
            reflect = ((i // 4) == 1)
            rot = i % 4
            s,a,sp = self_apply_transform(reflect, rot, s_arr, a_rr, sp_arr)

            new_s_arr[beg:end] = s
            new_a_arr[beg:end] = a
            new_r_arr[beg:end] = r_arr
            new_sp_arr[beg:end] = sp
            new_d_arr[beg:end] = d_arr
            new_v_arr[beg:end] = v_arr

        # Interlace the transforms, because we don't want to bias one orientation over another
        # (E.g. if buffer were of size 40, and we added 8 examples, the 8th orientation would never get put in the buffer)
        reord_new_s_arr = np.zeros((arr_len,) + self._s_shape)
        reord_new_a_arr = np.zeros((arr_len,))
        reord_new_r_arr = np.zeros((arr_len,))
        reord_new_sp_arr = np.zeros((arr_len,) + self._s_shape)
        reord_new_d_arr = np.zeros((arr_len,))
        reord_new_v_arr = np.zeros((arr_len,))

        for i in range(8):
            reord_new_s_arr = new_s_arr[i::8]
            reord_new_a_arr = new_a_arr[i::8]
            reord_new_r_arr = new_r_arr[i::8]
            reord_new_sp_arr = new_sp_arr[i::8]
            reord_new_d_arr = new_d_arr[i::8]
            reord_new_v_arr = new_v_arr[i::8]


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

        s_arr,a_arr,r_arr,sp_arr,done_mask,v_arr = self._exploit_symmetries(s_arr,a_arr,r_arr,sp_arr,done_mask,v_arr)
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
    rb = ReplayBuffer(3*8, (2,2), (1,), (1,))
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


