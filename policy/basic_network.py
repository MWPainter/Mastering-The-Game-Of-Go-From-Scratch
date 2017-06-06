import os
import gym
import numpy as np
import logging
import time
import sys
from network import QN
import gym
from gym import wrappers
from collections import deque

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.platform import gfile

from linear_schedule import LinearExploration, LinearSchedule

from util import get_logger, Progbar
from replay_buffer import ReplayBuffer

import configs


class BasicNetwork(QN):
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
        - self.lr: learning rate, type = float32
        """
        state_shape = self.board_shape
        self.s = tf.placeholder(name="state_input", dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,)) 
        self.r = tf.placeholder(dtype=tf.float32, shape=(None,)) 
        self.sp = tf.placeholder(dtype=tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2])) 
        self.done_mask = tf.placeholder(dtype=tf.bool, shape=(None,)) 
        self.lr = tf.placeholder(dtype=tf.float32) 



    def get_output_op(self, features_op, scope, reuse=False):
        """
        Get the tf op for the output of the networ
        If this is a Q-network, then it should be the Q-values
        If it's a p-network, then it should be the prob distr over actions
        THis function needs to take the features_op (the main part of the network) and then bolting 
        on another layer or two on the end

        Args:
            features_op: tf op for the first part of the network
            scope: scope to use with the network
            reuse: reuse variables
        """


    def add_update_target_op(self, scope, target_scope):
        """
        Returns the update target op
        When run it should copy the variables from inside scope 'scope' to scope 'target_scope'
        It's called periodically to update the target network
    
        Args:
            scope: name of the scope of variables in the network being trained
            target_scope: name of the scope of variables in the target network
        """
        var_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        target_var_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
        ops = []
        for (var,target_var) in zip(var_col,target_var_col):
          ops += [tf.assign(var, target_var_col)]
        self.update_target_op = tf.group(*ops)


    def add_loss_op(self, output_op, target_output_op):
        """
        Return tensorflow loss op
        For Q network, set (Q_target - Q)^2
        
        Args:
            output_op: Tf op for output from the network
            target_output_op: tf op for output from the target network
        """
        num_actions = self.num_actions
        q = target_op                       # use same logic from q learning
        target_q = target_output_op         # use same logic from q learning
        gamma = self.config.gamma
        mask = tf.logical_not(self.done_mask)

        adjust = gamma * tf.reduce_max(target_q, axis=(1,)) * tf.cast(mask, tf.float32)
        qsamp = self.r + adjust
        action_mask = tf.one_hot(self.a, num_actions)
        qsa = tf.boolean_mask(q, tf.cast(action_mask, tf.bool))
        self.loss = tf.reduce_mean((qsamp - qsa)**2)


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
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


"""
Some testing :)
"""
if __name__ == '__main__':
    # exploration strategy
    exp_schedule = LinearExploration(config.eps_begin, config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = BasicNetwork(5, configs.BasicNetworkTestConfig)
    model.run(exp_schedule, lr_schedule)






