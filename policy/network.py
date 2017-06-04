import os
import gym
import numpy as np
import logging
import time
import sys
from gym import wrappers
from collections import deque

from util import get_logger, Progbar
from replay_buffer import GoReplayBuffer


class N(object):
    """
    Abstract Class for implementing a network (for playing go)
    """
    def __init__(self, env_name, pachi_env_name, board_shape, config, logger=None):
        """
        Initialize network

        Args:
            env_name: name of the go self play environment
            pachi_env_name: name of the go pachi play environment
            board_shape: shape of the board, needed for replay buffers
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # Create the environments to use
        self.env = gym.make(env_name)
        self.pachi_env = gym.make(pachi_env_name)

        # Store the board (state) shape, action shape and reward shape
        self.board_shape = self.env.state.board.encode().shape
        self.action_shape = (1,)
        self.reward_shape = (1,)
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        # build model
        self.build()


    def build(self):
        """
        Build model
        """
        raise NotImplementedError

 
    def checkpoint(self, timestep):
        """
        Save a checkpoint of the current params
        Args:
            timestep: the time for which we are checkpointing
        """
        raise NotImplementedError


    def save(self):
        """
        Save the final model parameters
        """
        raise NotImplementedError


    def initialize(self):
        """
        Initialize variables if necessary
        """
        raise NotImplementedError


    def get_best_action(self, state):
        """
        Gets the best action for this state, and the distribution that was decided from
        Returns:
            tuple: best_action, action_distribution
        """
        raise NotImplementedError


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


    def _board_from_env(self, state, player):
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
        board = state.board.encode()
        if player != self.env.player_color:
            tmp = board[0]
            board[0] = board[1]
            board[1] = tmp
        return board


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.board_shape, self.action_shape, self.reward_shape)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_p_values = deque(maxlen=1000)
        p_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]
        
        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            states = []
            actions = []
            rewards_guess = []
            next_states = []
            while True:
                t += 1
                last_eval += 1
                last_record += 1

                if self.config.render_train: self.env.render()

                # Who's turn is it?
                player = self.env.state.color

                # chose action according to current state and exploration
                best_action, action_dist = self.get_best_action(state)
                action                   = exp_schedule.get_action(best_action)

                # store q values
                max_p_values.append(max(action_dist))
                p_values += list(action_dist)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # Store the s, a, new_s, for later use in replay buffer
                # Guessing the rewards, to be corrected when the game finishes
                states.append(self._board_from_state(state, player))
                actions.append(action)
                if t % 2 == 0: rewards_guess.append(1.0)
                else: rewards_guess.append(-1.0)
                next_states.append(self._board_from_state(new_state, player))

                # store the transition
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_p_values, p_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
                                        ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max Q", self.max_q), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done:
                    # Update replay buffer, first making sure the rewards (we guessed) are correct
                    # multiplying by (rewards[-1] * reward) is correct. If reward == 0 it zeros the array
                    # if rewards[-1] == reward, then it's 1, if rewards != reward, then it's -1
                    rewards = np.array(rewards_guess) * rewards_guess[-1] * reward
                    replay_buffer.store_go(states, actions, rewards, next_states)
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)          

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)


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
        if (t > self.config.learning_start and t % self.config.learning_freq == 0) and replay_buffer.should_sample():
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.checkpoint_freq == 0):
            self.checkpoint(t)

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.pachi_env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                # Play a step
                action = self.get_best_action(state)
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




def QN(N):
    """
    'Abstract' class encapsulating a network that will learn a policy function
    These networks are learned using reinforcement lerarning
    """
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

    def update_target_params(self):
        """
        Update params of Q' (target network) with params of Q
        """
        raise NotImplementedError




def PN(N):
    """
    'Abstract' class encapsulating a network that will learn a policy function
    These networks are learned using reinforcement learning
    """
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


    def update_target_params(self):
        """
        Update params of p' (target network) with params of p
        """
        raise NotImplementedError


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """
        pass



def VN(N):
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


    def train(self, exp_schedule, lr_schedule):
        raise NotImplementedError
    def train_step(self, t, replay_buffer, lr):
        raise NotImplementedError
    def evaluate(self, env=None, num_episodes=None):
        raise NotImplementedError
    def run(self, exp_schedule, lr_schedule):
        raise NotImplementedError
