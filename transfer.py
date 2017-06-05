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
        # GO shape is 3 x board_widht x board_width. SO need to move so channels are at back.
        self.s = tf.placeholder(name="state_input", dtype=tf.float32, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,)) 
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,)) 
        self.sp = tf.placeholder(dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
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
        input_channels = state_shape[0] 
        flat_input_size = state_shape[0]*state_shape[1]*state_shape[2]

        state = tf.transpose(self.s, [0,2,3,1]) # go inputs have channels first, need to transpose


        num_small_forwards = 1 # TODO IMPLEMENT AND TEST THIS
        small_model_input_shape = [-1, prev_board_width, prev_board_width, input_channels]
        small_model_input_size = small_model_input_shape[1]*small_model_input_shape[2]*small_model_input_shape[3]

        ###########################################################
        # Policy Convolution Network (same architecture as paper) #
        ###########################################################
        # this happens regardless of whether we are transfer learning or not.
        conv_layers_before_transfer = 3 
        total_conv_layers = 5 # total conv layers is 12 in paper (not including final 1x1 layer)
        k = 32 # this is 192 in the paper

        with tf.variable_scope(scope):
          x = state
          for i in range(conv_layers_before_transfer):
            kernel_size = 5 if i == 1 else 3
            with tf.variable_scope("layer_%d" % i):
              x = tf.layers.conv2d(
                            inputs=x, 
                            filters=k, 
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
          board_rep = x # this will be concatenated with board representations from
                        # the transfer learning, and then passed through the remainder
                        # of the convolutional layers

        if TRANSFER:
          ######################################
          # First, map large state to small in #
          ######################################

          # Method 1: convolution of small network
          pbw = prev_board_width; bw = board_width
          num_strides = bw - pbw + 1
          total_slices = num_strides**2
          # first, get slices of board state. 
          slices_start_indices = [(i,j) for i in range(num_strides) for j in range(num_strides)]
          slices = [tf.slice(state, [0,i,j,0], [-1, pbw, pbw, -1]) for (i,j) in slices_start_indices]
          # concatenate them so we can pass to small network in one batch
          small_input1 = tf.concat(slices, axis=0)
          small_input = tf.transpose(small_input1, [0,3,1,2])

          # this loads the small graph, and sets its input to be model1_in
          with gfile.FastGFile(prev_dir + '/frozen.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
          small_out = tf.import_graph_def(graph_def, input_map={"state_input:0" : small_input}, return_elements=['out:0'])[0]
          # has shape [batch_size * num_slices, num_actions]
          # small_out = tf.reshape(small_out, [-1, (prev_board_width*prev_board_width+2) * num_small_forwards])
          # small_out_size = (prev_board_width*prev_board_width+2)*num_small_forwards

          #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
          #print([n.name for n in tf.get_default_graph().as_graph_def().node])

          ######################################
          # now, map small out to to large out #
          ######################################

          # Method 1: convolution of small network
          # might be possible to vectorize all this with numpy indexing. TODO
          # unstack slices first
          slices = tf.split(small_out, total_slices, axis=0) # each slice has shape [batch_size, pbw**2]
          # ignore pass action and resign action, and reshape into shape of small board
          #slices = [s[:, :-2] for s in slices]
          slices = [tf.reshape(s, [-1, pbw, pbw]) for s in slices] # [batch_size, pbw, pbw] 
          # pad all the slices back to large board shape
          # TODO convert paddings to tensor?
          slices = [tf.pad(s, [[0,0],[i,bw-pbw-i],[j,bw-pbw-j]]) for (s,(i,j)) in zip(slices,slices_start_indices)]
          transfer_out1 = tf.stack(slices, axis=3) # has size [-1, bw, bw, total_slices]
          # concatenate this with normal board representation
          board_rep = tf.concat([transfer_out1, board_rep], axis=3)


          ###############################
          # METHOD 2: CONV+SMALL+DECONV #
          ###############################
          # this is going to be hard to get working with general layer sizes
          #conv_deconv_layers = 2
          #k = 32 

          #with tf.variable_scope(scope):
          #  x = state
          #  for i in range(conv_deconv_layers):
          #    #kernel_size = 5 if i == 1 else 3
          #    with tf.variable_scope("layer_%d" % i):
          #      x = tf.layers.conv2d(
          #                    inputs=x, 
          #                    filters=k, 
          #                    kernel_size=4,
          #                    strides=1,
          #                    padding='same',
          #                    activation=tf.nn.relu,
          #                    bias_initializer=tf.zeros_initializer())
          #  board_rep = x # this will be concatenated with board representations from
          #                # the transfer learning, and then passed through the remainder
          #                # of the convolutional layers




        # now, continue the policy network convolution on the board rep
        with tf.variable_scope(scope):
          x = board_rep
          for i in range(conv_layers_before_transfer, total_conv_layers):
            kernel_size = 5 if i == 1 else 3 # this line isn't really necessary here
            with tf.variable_scope("layer_%d" % i):
              x = tf.layers.conv2d(
                            inputs=x, 
                            filters=k, 
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
          # last layer: kernel_size 1, one filter, and different bias for each position (action)
          x = tf.layers.conv2d(
                        inputs=x, 
                        filters=1, 
                        kernel_size=1,
                        strides=1)
          # x now has shape [batch_size, board_width, board_width, 1]
          x = tf.reshape(x, [-1, board_width**2]) # now shape = [batch_size, board_width **2]
          # now apply different bias to each position
          last_bias = tf.get_variable(name="last_conv_b",dtype=tf.float32, shape=[board_width**2], initializer=tf.zeros_initializer())
          out = x + last_bias

        # need to name this operation so we can access it for transfer learning
        # when loading the graph from the save
        out = tf.identity(out, name='out')

        # NEVER SURRENDER
        out = tf.pad(out,[[0,0],[0,2]])

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
        for (q,t) in zip(q_col,target_col):
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
