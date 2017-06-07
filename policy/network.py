import os
import gym
import numpy as np
import logging
import time
import sys
from gym import wrappers
from collections import deque

from util import get_logger, Progbar
from replay_buffer import ReplayBuffer

import tensorflow as tf


class N(object):
    """
    Abstract Class for implementing a network (for playing go)
    """



    ###########################################
    ### Construction                        ###
    ###########################################

    def __init__(self, board_size, config, logger=None):
        """
        Initialize network

        Args:
            board_size: size of the board that this is going to be a network for
            config: class with hyperparameters
            scope: the scope under which the variables will be 
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # Create the environments to use (and reset to populate them with info)
        self.board_size = board_size
        #self.env = gym.make(self._self_play_env_name)
        self.env = gym.make(self._pachi_env_name)
        self.env.reset()
        self.pachi_env = gym.make(self._pachi_env_name)
        self.pachi_env.reset()

        # Store the board (state) shape, action shape and reward shape
        self.board_shape = self.env.state.board.encode().shape
        self.action_shape = ()
        self.reward_shape = ()
        self.num_actions = board_size ** 2
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        # build model
        self.build()


    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        state = self.process_state(self.s)
        self.outputs = self.get_output_op(state, scope=self.config.scope, reuse=False)

        # compute Q values of next state
        ###state_p = self.process_state(self.sp)
        ###self.target_outputs = self.get_output_op(state_p, scope=self.config.target_scope, reuse=False)

        # add update operator for target network
        self.add_update_target_op(self.config.scope, self.config.target_scope)

        # add square loss
        self.add_loss_op(self.outputs) ###, self.target_outputs)

        # add optmizer for the main networks
        self.add_optimizer_op(self.config.scope)


    def initialize(self):
        """
        Assumes the graph has been constructed (run time initialization) 
        This is the first thing that run calls
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise network and target network
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

        # checkpoint straight away, coz why not
        self.checkpoint(0)



    ###########################################
    ### Helpers                             ###
    ###########################################

    @property
    def _pachi_env_name(self):
        """
        Helper to grab the go env name
        """
        return 'Go' + str(self.board_size) \
                    + 'x' \
                    + str(self.board_size) \
                    + '-v0'


    @property
    def _self_play_env_name(self):
        """
        Helper to grab the self play env name
        """
        return 'SelfPlayGo' + str(self.board_size) \
                            + 'x' \
                            + str(self.board_size) \
                            + '-v0'


    def _board_from_player_perspective(self, state, player):
        """
        Helper to get the board (as a np.ndarray with shape (3,s,s) if s is the size of the board)
        the encapsulates the game state. Used in train below

        This is more complicated than it seems at first. Because board[0] is the player 1's peices and 
        board[1] is player 2's peices. We need to use information from both player 1 and player 2, but 
        pass it into the same format (board[0] being the agents peices and board[1] being the opponent 
        peices). Thus, when it's player 2's turn, we swap the peices.

        Furthermore, we cannot just use state.color, because otherwise in a (s,a,r,sp) example, we would 
        flip the peices on one of s and sp, and not the other

        Args:
            state: A go state from the go environment
            player: The player who's turn it is (self.env.state.color, *before* the action was performed)
        Returns:
            board: A np.ndarray with shape (3,s,s), where board[0] is the agents peices
        """
        board = state #.board.encode()     # apparently states returned from env.reset and env.step are the np arrays
        if player != self.env.player_color:
            tmp = board[0]
            board[0] = board[1]
            board[1] = tmp
        return board


    # Helper for get valid actions from an environment
    def _get_valid_action_indices(self, state):
        """
        Get's all of the open board positions.
        Converts those coordinates to action numbers
        Returns that lis
    
        Args:
            state: np.array of shape (3,board_size,board_size) for the state
        Returns:
            List of valid actions that an agent could take
        """
        # Get open board position
        free_spaces = state[2]
        non_zero_coords = np.transpose(np.nonzero(free_spaces))

        # Get action numbers
        non_zero_coords[:,0] *= self.config.board_size
        actions = np.sum(non_zero_coords, axis=1)

        # Return
        return actions




    ###########################################
    ### Defining tensorflow ops             ###
    ###########################################
        
    def add_placeholders_op(self):
        """
        Add tf placeholders to the class
        """
        raise NotImplementedError


    def get_output_op(self, features_op, scope, reuse=False):
        """
        Get the tf op for the output of the networ
        If this is a Q-network, then it should be the Q-values
        If it's a p-network, then it should be the prob distr over actions
        THis function needs to take the features_op (the main part of the network) and then bolting 
        on another layer or two on the end

        This must cr

        Args:
            state: The state (input to network)
            scope: scope to use with the network
            reuse: reuse variables
        """
        raise NotImplementedError


    def add_update_target_op(self, scope, target_scope):
        """
        Returns the update target op
        When run it should copy the variables from inside scope 'scope' to scope 'target_scope'
        It's called periodically to update the target network
    
        Args:
            scope: name of the scope of variables in the network being trained
            target_scope: name of the scope of variables in the target network
        """
        raise NotImplementedError


    def add_loss_op(self, output_op, target_output_op):
        """
        Return tensorflow loss op
        For Q network, set (Q_target - Q)^2
        
        Args:
            output_op: Tf op for output from the network
            target_output_op: tf op for output from the target network
        """
        raise NotImplementedError


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
        """
        state = tf.cast(state, tf.float32)

        return state





    ###########################################
    ### Saving models                       ###
    ###########################################
 
    def checkpoint(self, timestep):
        """
        Save a checkpoint of the current params
        Args:
            timestep: the time for which we are checkpointing
        """
        if not os.path.exists(self.config.model_checkpoint_output):
            os.makedirs(self.config.model_checkpoint_output)
        self.saver.save(self.sess, self.config.model_checkpoint_output + str(timestep))


    def save(self):
        """
        Save the final model parameters
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)
        self.saver.save(self.sess, self.config.model_output)





    ###########################################
    ### tensor board                        ###
    ###########################################

    def init_averages(self):
        """
        Extra attributes for tensorboard
        """
        self.avg_reward = -1.
        self.max_reward = -1.
        self.std_reward = 0

        self.avg_p = 0
        self.max_p = 0
        self.std_p = 0
        
        self.eval_reward = -1.


    def update_averages(self, rewards, max_p_values, p_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_p_values: deque
            p_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_p      = np.mean(max_p_values)
        self.avg_p      = np.mean(p_values)
        self.std_p      = np.sqrt(np.var(p_values) / len(p_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]


    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_p_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_p")
        self.max_p_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_p")
        self.std_p_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_p")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg p", self.avg_p_placeholder)
        tf.summary.scalar("Max p", self.max_p_placeholder)
        tf.summary.scalar("Std p", self.std_p_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)





    ###########################################
    ### Network interaction                 ###
    ###########################################

    @property
    def policy(self, state):
        """
        Returns a probability distribution function, taking states and returning probabilities over actions
        """
        return lambda state: self.get_action_distribution(state)


    def get_action_distribution(self, state):
        """
        Returns a distribution over actions 
    
        Args:
            state: tf variables for the current state of the game
        Returns:
            Probability distribution over actions         
        """
        raise NotImplementedError


    def get_best_valid_action(self, state):
        """
        Return best action (its the same regardless of if the function learned is p or Q)

        Args:
            state: the state to get the best (valid) action for
        Returns:
            action: (int) the best action
            action_values: (np array) q/p values for all actions
            valid_actions: (array) of indices that are valid
        """
        valid_actions = self._get_valid_action_indices(state)
        action_values = self.sess.run(self.outputs, feed_dict={self.s: [state]})[0]
        best_action_idx = np.argmax(action_values[valid_actions])      # get index into valid_actions of the best action to take
        return valid_actions[best_action_idx], action_values, valid_actions
    

    def update_target_params(self):
        """
        Update params of target network (just runs the op)
        """
        self.sess.run(self.update_target_op)


    ###########################################
    ### Training loop + eval                ###
    ###########################################

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
                Instance of LinearExploration
            lr_schedule: Schedule for learning rate
                Instance of LinearExploration
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.board_shape, self.action_shape, self.reward_shape)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_p_values = deque(maxlen=1000)
        p_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate(t)]
        
        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            states = []
            actions = []
            episode_rewards = []
            next_states = []
            done_mask = []
            values_guess = []
            while True:
                t += 1
                last_eval += 1
                last_record += 1

                if self.config.render_train: self.env.render()

                # Who's turn is it?
                player = self.env.state.color

                # chose action according to current state and exploration
                player_perspective_board = self._board_from_player_perspective(state,player)
                best_action, action_dist, valid_actions = self.get_best_valid_action(player_perspective_board)
                action                                  = exp_schedule.get_action(best_action, valid_actions)

                # store q values
                max_p_values.append(max(action_dist))
                p_values += list(action_dist)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # Store the s, a, new_s, for later use in replay buffer
                # Guessing the rewards, to be corrected when the game finishes
                states.append(self._board_from_player_perspective(state, player))
                actions.append(action)
                rewards.append(reward)
                episode_rewards.append(reward)
                next_states.append(self._board_from_player_perspective(new_state, player))
                if done: done_mask.append(1.0)
                else: done_mask.append(0.0)
                if t % 2 == 0: values_guess.append(1.0)
                else: values_guess.append(-1.0)

                # store the transition
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # Update schedules
                exp_schedule.update(t)
                lr_schedule.update(t)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0)):
                    self.update_averages(rewards, max_p_values, p_values, scores_eval)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
                                        ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max P", self.max_p), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done:
                    # Update replay buffer, first making sure the values (we guessed) are correct
                    # multiplying by (values[-1] * reward) is correct. If reward == 0 it zeros the array
                    # if values[-1] == reward, then it's 1, if values[-1] != reward, then it's -1
                    values = np.array(values_guess) * values_guess[-1] * reward
                    discounts = np.array([self.config.gamma ** i for i in xrange(values.shape[0])])
                    discounted_values = values * discounts
                    replay_buffer.store_example_batch(states, actions, episode_rewards, next_states, done_mask, discounted_values)
                    break

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate(t)]

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate(t)]


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if t > self.config.learning_start and replay_buffer.should_sample:
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.checkpoint_freq == 0):
            self.checkpoint(t)

        return loss_eval, grad_eval


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        ###s_batch, a_batch, r_batch, sp_batch, done_mask_batch, _ = replay_buffer.sample(self.config.batch_size)
        s_batch, a_batch, _, _, _, v_batch = replay_buffer.sample(self.config.batch_size)


        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            ###self.r: r_batch,
            ###self.sp: sp_batch, 
            ###self.done_mask: done_mask_batch,
            self.v: v_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: self.avg_reward, 
            self.max_reward_placeholder: self.max_reward, 
            self.std_reward_placeholder: self.std_reward, 
            self.avg_p_placeholder: self.avg_p, 
            self.max_p_placeholder: self.max_p, 
            self.std_p_placeholder: self.std_p, 
            self.eval_reward_placeholder: self.eval_reward, 
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval


    def save_opponent(self, some_scope_thing):
        """
        Save the current params to be able to be used as an opponent later
        """
        # TODO
        raise NotImplementedError


    def evaluate(self, t, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training

        Args:
            t: timestep
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.pachi_env

        state = env.reset()
        _, action_dist, _ = self.get_best_valid_action(state)
        print("Heatmap for initial state, at time " + str(t))
        print(action_dist.reshape((self.config.board_size,self.config.board_size)))

        # replay memory to play
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                # Play a step
                action, _, va = self.get_best_valid_action(state)
                new_state, reward, done, info = env.step(action)
                state = new_state

                # Count rewards
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)     

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward




    ###########################################
    ### Run/coord training                  ###
    ###########################################

    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()




    



class VN(N):
    """
    Class encapsulating a network that will learn a value function
    These networks are learned using supervised learning/regression, given data from self play with a 
    Q network or a policy network
    """
    def __init__(self, env, config, logger=None):
        # Give it a copy of PART of a netowork in constructor
        # Add two fully connected layers
        # TODO: Something encapsulating that we load ing data from a policy network
        # Probably pass in a policy network into the train for this
        # Furthermore, use that policy network to INITIALIZE variable in this network
        # Should be an INDEPENDENT copy, and not share variables
        pass

    # Functions that should be implemented
    def init_averages(self):
        raise NotImplementedError
    def update_averages(self, rewards, max_p_values, p_values, scores_eval):
        raise NotImplementedError
    def train(self, exp_schedule, lr_schedule):
        raise NotImplementedError
    def train_step(self, t, replay_buffer, lr):
        raise NotImplementedError
    def evaluate(self, env=None, num_episodes=None):
        raise NotImplementedError
    def run(self, exp_schedule, lr_schedule):
        raise NotImplementedError
    
    # Functions that maybe should be implemented?
    def update_target_params(self):
        pass

    # Functions that shouldn't be implemented
    def get_best_valid_action(self, state):
        pass
    def policy(self, state):
        pass
    def get_action_distribution(self, state):
        pass
        
