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
from basic_network import BasicNetwork

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

from linear_schedule import LinearExploration, LinearSchedule

from util import get_logger, Progbar
from replay_buffer import ReplayBuffer

import policy.configs





class CompositeNetwork(BasicNetwork):


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
        - self.v: batch of value functions (sum of *discounted* rewards)
        - self.lr: learning rate, type = float32
        """
        state_shape = self.board_shape
        self.s = tf.placeholder(name="state_input", dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,)) 
        self.v = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.lr = tf.placeholder(dtype=tf.float32) 


    def get_opponent_out(self, opponent_f):
        """
        Return an op for a graph with old weights (from freezing it)

        Args:
            opponent_f: the path to the frozen graph to turn into an opponent
        """
        with gfile.FastGFile(opponent_f, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, input_map={"state_input:0" : self.s}, return_elements=['out:0'])[0]


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
        Returns:
            output: a tf op, that when given a state and run, will output a probability distribution over actions
            logits: a tf op, returning the logits passed to the sofmax 
        """
        # initialize variables
        num_actions = self.num_actions
        state_shape = list(self.env.observation_space.shape)
        input_channels = state_shape[0] 
        flat_input_size = state_shape[0]*state_shape[1]*state_shape[2]

        # go states are (3, n, n), where 3 is number of channels
        # so state is (batch_size, 3, n, n)
        # but really we want it to be (batch_size, n, n, 3) for conv2d layers
        state = tf.transpose(state, [0,2,3,1]) # go inputs have channels first, need to transpose

        # Build a base cnn with 5 layers, and 32 filters
        board_rep = self._build_pure_convolution(inpt=state, num_layers=5, num_filters=32, scope=scope)
               
        # Build the output layers (1x1 convolution + softmax)
        output, logits = self._add_output_layers(inpt=board_rep, scope=scope)

        # need to name this operation so we can access it for transfer learning
        # when loading the graph from the save
        # Therefore this must *necessarily* be called 'out'
        output = tf.identity(output, name ='out')

        # export the meta graph now, so that it doesn't include optimizer variables
        graph = tf.get_default_graph()
        tf.train.write_graph(graph, self.config.output_path, self.config.graph_name)

        return output, logits


    def _build_pure_convolution(self, inpt, num_layers, num_filters, scope):
        """
        Construct a convolutional neural network, with output the same shape as the input
        First layer is a 5x5 convolution with stride of 1
        All subsequent layers are a 3x3 convolution with stride of 1

        Args:
            inpt: tf op for the input to the convolutional layers
            num_layers: the number of layers we should build in the CNN
            num_filters: the number of filters (in all layers) that we should use
            scope: tf variable scope to use
        Returns:
            The next layer to 
        """

        # the below makes a representation of the board by passing it through 
        # some convolutional layers. This will be stacked with representations
        # passed through the transfer learning layers, and then passed through
        # a final convolutional network.
        next_layer = inpt
        with tf.variable_scope(scope):
            for i in range(num_layers):
                kernel_size = 5 if i == 0 else 3
                with tf.variable_scope("layer_%d" % i):
                    next_layer = layers.conv2d(
                                            inputs=next_layer,
                                            kernel_size=kernel_size,
                                            num_outputs=num_filters,
                                            stride=1,
                                            padding='SAME',
                                            activation_fn=tf.nn.relu,
                                            biases_initializer=tf.zeros_initializer)
        board_rep = next_layer

        # Transfer learning
        ######################################
        # First, map large state to small in #
        ######################################

        # Method 1: convolution of small network
        pbw = self.config.transfer_board_width; bw = self.config.board_size
        num_strides = bw - pbw + 1
        total_slices = num_strides**2
        # first, get slices of board state. 
        slices_start_indices = [(i,j) for i in range(num_strides) for j in range(num_strides)]
        slices = [tf.slice(inpt, [0,i,j,0], [-1, pbw, pbw, -1]) for (i,j) in slices_start_indices]
        # concatenate them so we can pass to small network in one batch
        small_input1 = tf.concat(slices, axis=0)
        small_input = tf.transpose(small_input1, [0,3,1,2])
        small_input = tf.sigmoid(small_input)
        small_input = tf.cast(small_input, tf.uint8)

        # this loads the small graph, and sets its input to be model1_in
        with gfile.FastGFile(self.config.transfer_model_f, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
        small_out = tf.import_graph_def(graph_def, input_map={"state_input:0" : small_input}, return_elements=['out:0'])[0]
        # has shape [batch_size * num_slices, num_actions]
        # small_out = tf.reshape(small_out, [-1, (pbw*pbw+2) * num_small_forwards])
        # small_out_size = (pbw*pbw+2)*num_small_forwards

        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print([n.name for n in tf.get_default_graph().as_graph_def().node])

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
        # FOR NOW, ASSUME THAT bw = pbw + 2*n for some integer n
        conv_deconv_start_layers = 2 # number of convolution layers with SAME padding before VALID reduction
        k = self.config.num_filters_global

        with tf.variable_scope(scope):
          x = inpt
          for i in range(conv_deconv_start_layers):
            #kernel_size = 5 if i == 1 else 3
            with tf.variable_scope("cd_conv_layer_%d" % i):
              x = tf.layers.conv2d(
                            inputs=x, 
                            filters=k, 
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
          # every convolution with kernel_size=3, stride=1, and padding='valid' will reduce bw by 2
          # so we need to reduce until size is equal to pbw
          for i in range(bw, pbw, -2):
            with tf.variable_scope("cd_red_layer_%d" % i): # "red" for "reduction"
              if i == pbw + 2: # the last layer
                k = 3 # the number of channels for board state
              x = tf.layers.conv2d(
                            inputs=x, 
                            filters=k, 
                            kernel_size=3,
                            strides=1,
                            padding='valid',
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
          # x has size [-1, pbw, pbw, 3]
          #print("X SHAPE")
          #print(x.get_shape())

        small_input = tf.transpose(x, [0,3,1,2])
        small_input = tf.sigmoid(small_input)
        small_input = tf.cast(small_input, tf.uint8)
        small_features = tf.import_graph_def(graph_def, input_map={"state_input:0" : small_input}, return_elements=[config.transfer_features + ':0'])[0]
        with tf.variable_scope(scope):
          # every convolution with kernel_size=3, stride=1, and padding='valid' will reduce bw by 2
          # so we need to reduce until size is equal to pbw
          x = small_features
          for i in range(pbw, bw, 2):
            with tf.variable_scope("cd_deconv_layer_%d" % i): # "red" for "reduction"
              # should we do this?
              #if i == bw - 2: # the last layer
              #  k = 3 # the number of channels for board state
              x = tf.layers.conv2d_transpose(
                            inputs=x, 
                            filters=k, 
                            kernel_size=3,
                            strides=1,
                            padding='valid',
                            activation=tf.nn.relu,
                            bias_initializer=tf.zeros_initializer())
          

            
            
          board_rep = tf.concat([x, board_rep], axis=3)

        # now, continue the policy network convolution on the board rep
        with tf.variable_scope(scope):
          x = board_rep
          for i in range(self.config.conv_layers_before_transfer, self.config.total_conv_layers):
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
               
        x = tf.identity(x, name='final_features') # for transfer learning

        return x


    def _add_output_layers(self, inpt, scope):
        """
        Adds a 1x1 convolution and a linear activation to compute logits
        We have a kernel of size 1, with stride of 1, each with a seperate bias
        Passes logits through a softmax activation

        Args:
            inpt: the output from previous layers in the network (the features)
            scope: tf variable scope to use
        Returns:
            probs: tf op that will evaluate to a probability distribution over actions
            logits: the tf op input to the softmax activation, to be used in further (composite) parts of the network
        """
        logits = None
        out = None
        with tf.variable_scope(scope):
            # last layer: kernel_size 1, one filter, and different bias for each position (action)
            x = layers.conv2d(
                            inputs=inpt, 
                            num_outputs=1, 
                            kernel_size=1,
                            stride=1)
            # x has shape [batch_size, bw, bw, 1], so reshape to [batch_size, bw**2]
            x = tf.reshape(x, [-1, self.config.board_size**2]) 
          
            # apply different bias to each position
            bias = tf.get_variable(name="last_conv_b",dtype=tf.float32, shape=[self.config.board_size**2], initializer=tf.zeros_initializer)
            logits = x + bias
    
            # apply softmax
            probs = tf.nn.softmax(logits)
            out = probs
        return out, logits
        

    def add_loss_op(self, output_op): 
        """
        This implements a loss that will lead to updates with terms of the correct form for policy gradients

        Args:
            output_op: Tf op for output from the network
            target_output_op: tf op for output from the target network
        """
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
    config = policy.configs.compositeConfig

    # exploration strategy
    exp_schedule = LinearExploration(config.eps_begin, config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = CompositeNetwork(config)
    model.run(exp_schedule, lr_schedule)






