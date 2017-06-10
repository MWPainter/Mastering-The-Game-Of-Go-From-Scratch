import os
import gym
import numpy as np
import logging
import time
import sys
from network import N
import gym
from gym import wrappers
from collections import deque

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

from linear_schedule import LinearExploration, LinearSchedule

from util import get_logger, Progbar
from replay_buffer import ReplayBuffer

import policy.configs


class BasicNetwork(N):
    """
    Plain ol' CNN to play a (very) small game of go
    """


    ###########################################
    ### Defining tensorflow ops             ###
    ###########################################
        
    def add_placeholders_op(self):
        """
        Add tf placeholders to the class, namely:

        - self.s: batch of states, type = uint8
                  shape = (batch_size, board_size, board_size, 3)
        - self.a: batch of actions, type = int32
                  shape = (batch_size)
        - self.r: batch of rewards, type = float32
                  shape = (batch_size)
        - self.sp: batch of next states, type = uint8
                   shape = (batch_size, board_size, board_size, 3)
        - self.done_mask: batch of done, type = bool
                          shape = (batch_size)
                          note that this placeholder contains bool = True only if we are 
                          done in the relevant transition
        - self.v: batch of value functions (sum of *discounted* rewards)
        - self.lr: learning rate, type = float32
        """
        state_shape = self.board_shape
        self.s = tf.placeholder(name="state_input", dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,)) 
        ###self.r = tf.placeholder(dtype=tf.float32, shape=(None,)) 
        ###self.sp = tf.placeholder(dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        ###self.done_mask = tf.placeholder(dtype=tf.bool, shape=(None,)) 
        self.v = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.lr = tf.placeholder(dtype=tf.float32) 



    def update_opponent(self, opponent_f):
        with gfile.FastGFile(opponent_f, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
        self.opponent_out = tf.import_graph_def(graph_def, input_map={"state_input:0" : self.s}, return_elements=['out:0'])[0]

    def get_output_op(self, state, scope, reuse=False):
        """
        Get the tf op for the output of the networ
        If this is a Q-network, then it should be the Q-values
        If it's a p-network, then it should be the prob distr over actions
        THis function needs to take the features_op (the main part of the network) and then bolting 
        on another layer or two on the end

        Args:
            state: tf variable for the input to the network (i.e. the state is the input to the network)
            scope: scope to use with the network
            reuse: reuse variables
        """
        # this information might be useful
        num_actions = self.num_actions
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

        state = tf.transpose(state, [0,2,3,1]) # go inputs have channels first, need to transpose


        num_small_forwards = 1 # TODO IMPLEMENT AND TEST THIS

        ###########################################################
        # Policy Convolution Network (same architecture as paper) #
        ###########################################################
        # this happens regardless of whether we are transfer learning or not.
        conv_layers_before_transfer = 3 
        total_conv_layers = 5 # total conv layers is 12 in paper (not including final 1x1 layer)
        k = 32 # this is 192 in the paper

        board_rep = None

        with tf.variable_scope(scope):
          x = state
          for i in range(conv_layers_before_transfer):
            kernel_size = 5 if i == 1 else 3
            with tf.variable_scope("layer_%d" % i):
              x = layers.conv2d(
                            inputs=x, 
                            num_outputs=k, 
                            kernel_size=kernel_size,
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.zeros_initializer)
          board_rep = x # this will be concatenated with board representations from
                        # the transfer learning, and then passed through the remainder
                        # of the convolutional layers


        # now, continue the policy network convolution on the board rep
        with tf.variable_scope(scope):
          x = board_rep
          for i in range(conv_layers_before_transfer, total_conv_layers):
            kernel_size = 5 if i == 1 else 3 # this line isn't really necessary here
            with tf.variable_scope("layer_%d" % i):
              x = layers.conv2d(
                            inputs=x, 
                            num_outputs=k, 
                            kernel_size=kernel_size,
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.zeros_initializer)
               
        x = tf.identity(x, name='final_features') # for transfer learning
        with tf.variable_scope(scope):
          # last layer: kernel_size 1, one filter, and different bias for each position (action)
          x = layers.conv2d(
                        inputs=x, 
                        num_outputs=1, 
                        kernel_size=1,
                        stride=1)
          # x now has shape [batch_size, board_width, board_width, 1]
          x = tf.reshape(x, [-1, self.config.board_size**2]) # now shape = [batch_size, board_width **2]
          # now apply different bias to each position
          last_bias = tf.get_variable(name="last_conv_b",dtype=tf.float32, shape=[self.config.board_size**2], initializer=tf.zeros_initializer)
          logits = x + last_bias

          # apply softmax
          probs = tf.nn.softmax(logits)
          out = probs

        # need to name this operation so we can access it for transfer learning
        # when loading the graph from the save
        out = tf.identity(out, name ='out')

        # NEVER SURRENDER
        #out = tf.pad(out,[[0,0],[0,2]])

        # export the meta graph now, so that it doesn't include optimizer variables
        if scope==self.config.scope:
          graph = tf.get_default_graph()
          tf.train.write_graph(graph, self.config.output_path, self.config.graph_name)

        ##############################################################
        ######################## END YOUR CODE #######################

        return out





    def add_update_target_op(self, scope, target_scope):
        """
        Returns the update target op
        When run it should copy the variables from inside scope 'scope' to scope 'target_scope'
        It's called periodically to update the target network
    
        Args:
            scope: name of the scope of variables in the network being trained
            target_scope: name of the scope of variables in the target network
        """
        varss = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
        ops = []
        for (v,tv) in zip(varss,target_vars):
            ops += [tf.assign(tv, v)]
        self.update_target_op = tf.group(*ops)


    def add_loss_op(self, output_op): ###, target_output_op):
        """
        Return tensorflow loss op
        For Q network, set (Q_target - Q)^2
        
        Args:
            output_op: Tf op for output from the network
            target_output_op: tf op for output from the target network
        """
        ###num_actions = self.num_actions
        ###q = output_op                       # use same logic from q learning
        ###target_q = target_output_op         # use same logic from q learning
        ###gamma = self.config.gamma
        ###mask = tf.logical_not(self.done_mask)

        ###adjust = gamma * tf.reduce_max(target_q, axis=(1,)) * tf.cast(mask, tf.float32)
        ###qsamp = self.r + adjust
        ###action_mask = tf.one_hot(self.a, num_actions)
        ###qsa = tf.boolean_mask(q, tf.cast(action_mask, tf.bool))
        ###self.loss = tf.reduce_mean((qsamp - qsa)**2)

        action_mask = tf.one_hot(self.a, self.num_actions)
        output_sa = tf.boolean_mask(output_op, tf.cast(action_mask, tf.bool)) # set output for s,a for the batch

        self.loss = tf.reduce_mean(-tf.log(output_sa + 1e-07) * self.v)




    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
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


"""
Some testing :)
"""
if __name__ == '__main__':
    # Grab the test config
    config = policy.configs.bntconfig

    # exploration strategy
    exp_schedule = LinearExploration(config.eps_begin, config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = BasicNetwork(config.board_size, config)
    model.run(exp_schedule, lr_schedule)






