import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

import gym

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.transfer import config



TRANSFER = config.transfer
board_width = config.board_size
prev_board_width = config.prev_board_size
save_dir = config.output_path
prev_dir = config.prev_output_path

class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        # NOTE: was originally tf.uint8, I made a float so could be set with output of other model
        self.s = tf.placeholder(name="state_input", dtype=tf.float32, shape=(None, state_shape[0], state_shape[1], state_shape[2]*self.config.state_history)) 
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,)) 
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,)) 
        self.sp = tf.placeholder(dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2]*self.config.state_history)) 
        self.done_mask = tf.placeholder(dtype=tf.bool, shape=(None,)) 
        self.lr = tf.placeholder(dtype=tf.float32) 

        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                W s where W is a matrix of shape m x n

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        state_shape = list(self.env.observation_space.shape)
        flat_input_size = state_shape[0]*state_shape[1]*state_shape[2]*self.config.state_history
        #flat_input_size = state*board_width*state_shape[2]*self.config.state_history

        num_small_forwards = 1 # TODO test changing this thing
        small_model_input_shape = [-1, prev_board_width, prev_board_width, state_shape[2]*self.config.state_history]
        # TODO IDK why this is the input shape for the 5x5 model, investigate this
        small_model_input_shape = [-1, 3, 5, 20]

        small_model_input_size = small_model_input_shape[1]*small_model_input_shape[2]*small_model_input_shape[3]


        # THIS WASN'T WORKING, so I just made original placeholder a float. 
        # probably not the best way to do it
        #float_state = tf.cast(self.s, tf.float32, name="float_state")

        if TRANSFER:
          # FIRST MAP FROM LARGE TO SMALL BOARD

          # TODO is this still true?
          # NOTE: took me forever to figure this out. Can't have "import_meta_graph" within a "tf.variable_scope" or else everything is screwed up. 
          # so need to only use this scoping around the actual variable definitions.
          with tf.variable_scope(scope):
            W1 = tf.get_variable(name="W_to_small",dtype=tf.float32, shape=[flat_input_size, small_model_input_size], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name="b_to_small",dtype=tf.float32, shape=[small_model_input_size], initializer=tf.zeros_initializer())
          small_input = tf.matmul(tf.contrib.layers.flatten(self.s), W1) + b1 

          small_input = tf.reshape(small_input, small_model_input_shape)

          # this loads the small graph, and sets its input to be model1_in
          with gfile.FastGFile(prev_dir + '/frozen.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
          small_out = tf.import_graph_def(graph_def, input_map={"state_input:0" : small_input}, return_elements=['out:0'])
          small_out = tf.reshape(small_out, [-1, (prev_board_width*prev_board_width+2) * num_small_forwards])
          small_out_size = (prev_board_width*prev_board_width+2)*num_small_forwards

          #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
          #print([n.name for n in tf.get_default_graph().as_graph_def().node])

          # NOW, MAP SMALL OUT TO LARGE OUT
          with tf.variable_scope(scope):
            W2 = tf.get_variable(name="W_to_out",dtype=tf.float32, shape=[small_out_size, num_actions], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name="b_to_out",dtype=tf.float32, shape=[num_actions], initializer=tf.zeros_initializer())
          out = tf.matmul(small_out, W2) + b2 

        else:
          with tf.variable_scope(scope):
            W1 = tf.get_variable(name="W1",dtype=tf.float32, shape=[flat_input_size, num_actions], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name="b1",dtype=tf.float32, shape=[self.env.action_space.n], initializer=tf.zeros_initializer())
          out = tf.matmul(tf.contrib.layers.flatten(self.s), W1) + b1 

        out = tf.identity(out, name='out')

        # export the meta graph now, so that it doesn't include optimizer variables
        if scope=='q':
          graph = tf.get_default_graph()
          tf.train.write_graph(graph, save_dir, 'graph_save.pb')

        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        q_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        ops = []
        print(q_col)
        print(target_col)
        for (q,t) in zip(q_col,target_col):
          print(q)
          print(t)
          ops += [tf.assign(t, q)]
        self.update_target_op = tf.group(*ops)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        gamma = self.config.gamma
        mask = tf.logical_not(self.done_mask)

        adjust = gamma * tf.reduce_max(target_q, axis=(1,)) * tf.cast(mask, tf.float32)
        qsamp = self.r + adjust
        action_mask = tf.one_hot(self.a, num_actions)
        qsa = tf.boolean_mask(q, tf.cast(action_mask, tf.bool))
        self.loss = tf.reduce_mean((qsamp - qsa)**2)


        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_var_list = optimizer.compute_gradients(self.loss, variables)
        grads, variables = zip(*grads_var_list)
        if self.config.grad_clip:
                grads_new = []
                for g in grads:
                        if not g is None:
                                grads_new += [tf.clip_by_norm(g, self.config.clip_val)]
                        else:
                                grads_new += [g]
                grads = grads_new
        grads_var_list = zip(grads, variables)
        self.train_op = optimizer.apply_gradients(grads_var_list)
        self.grad_norm = tf.global_norm(grads)
        
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
 
    env = gym.make('Go%dx%d-v0' % (board_width, board_width))

    # exploration strategy
    config.eps_nsteps=100
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    config.lr_nsteps=100
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
