import gym
import copy
import tensorflow as tf
from tensorflow.python.platform import gfile



def _opponent(player):
    """
    Return opponent color to the supplied playe number
    Args:
        Player: either a 1 or 2, 1 for black, 2 for white
    """
    if player == 1:
        return 2
    else:
        return 1

class GoGameState(object):
    """
    Wrapper for the go game state, to be used by the monte carlo tree sreach
    It just implements all of the methods required by the MCTS, as described in MCTS/mcts.py
    """
    def __init__(self, board_size, env=None):
        if env == None:
            env_name = "SelfPlayGo%dx%d-v0" % (board_size, board_size)
            self.env = gym.make(env_name)
        else:
            self.env = copy.copy(env)
        self.board_size = board_size
        self.game_ended = False
        self.winner = None
        self.player = self.env.state.color


    def get_actions(self):
        return range(self.board_size ** 2 + 1)


    def winner(self):
        return self.winner


    def succ(self, action):
        next_state = GoGameState(self.board_size, self.env)
        player = self.env.state.color # note that this would be wrong if move down a line,
                                      # reward is from this players perspective
        _, reward, done, _ = next_state.env.step(action)
        if done:
            if reward < 0:
                next_state.winner = _opponent(player)
            elif reward > 0:
                next_state.winner = player
        return next_state


    def utility(self, player):
        if self.winner == player:
            return 1.0
        elif self.winner == None:
            return 0.0
        else:
            return -1.0


    @property
    def default_action(self):
        return self.board_size ** 2 


    @property
    def resign_move(self):
        return self.board_size ** 2 + 1


    @property
    def state(self):
        return self.env.state.board.encode()






class NetworkPolicy(object):
    """
    Wrapper for a network policy, needs to provide a function that takes game states and 
    returns a probability distribution
    """
    def __init__(self, network_filename, board_size):
        """
        Args:
            network_filename: the filename of the trained network to use as a policy
            board_size: size of the board that this network is for (probably get this from a config)
        """
        self.s = tf.placeholder(name="state_input", dtype=tf.uint8, shape=(None, 3, board_size, board_size))
        with gfile.FastGFile(network_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.network = tf.import_graph_def(graph_def, input_map={"state_input:0" : self.s}, return_elements=['out:0'])[0]
        self.sess = tf.Session()
        


    def policy(self, game_state):
        """
        Return the policy for the given game state!

        Args:
            game_state: The GoGameState to return a policy for
        """
        state = game_state.state
        return self.sess.run(self.network, feed_dict={self.s: [state]})[0]


class GreedyNetworkAgent(object):
    def __init__(self, network_filename, board_size):
        self.network_policy = NetworkPolicy(network_filename, board_size)

    def get_action(self, game_state):
        return np.argmax(self.network_policy.policy(game_state))


class MCTSNetworkAgent(object):
    """
    Wrapper for network + mcts agent
    """
    def __init__(self, network_filename, board_size, num_iters=5000):
        self.network_policy = NetworkPolicy(network_filename, board_size)
        policy_fn = self.network_policy.policy
        self.mcts_agent = MCTreeSearchAgent(policy_fn, policy_fn, iter=num_iters)

    def get_action(self, game_state):
        return self.mcts_agent.getAction(game_state)




class Eval(object):
    """
    Evaluates two agents
    Specifies by 'Pachi' or the filename, and if MCTS should be used with the policy
    """
    def __init__(self, agent1_name, agent1_mcts, agent2_name, agent2_mcts, board_size, results_filename, 
                 num_games=100):
        self.board_size = board_size
        self.results_filename = results_filename
        self.num_games = num_games

        if not os.path.exists(self.results_filename):
            os.makedirs(self.results_filename)

        if agent2_name == 'Pachi':
            agent_tmp = agent1_name
            agent1_name = agent2_name
            agent2_name = agent_tmp
            agentTmp = agent1_mcts
            agent1_mcts = agent2_mcts
            agent2_mcts = agent_tmp
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.agent1 = None

        if agent1_name != 'Pachi':
            if agent1_mcts:
                self.agent1 = MCTSNetworkAgent(agent1_name, board_size)
            else: 
                self.agent1 = GreedyNetworkAgent(agent1_name, board_size)

        if agent2_mcts:
            self.agent2 = MCTSNetworkAgent(agent2_name, board_size)
        else: 
            self.agent2 = GreedyNetworkAgent(agen2_name, board_size)

        env_name = "Go%dx%d-v0" % (board_size, board_size)
        if self.agent1 != None:
            env_name = "SelfPlay" + env_name
        self.env = gym.make(env_name)


    def evaluate(self):
        if self.agent1 == None:
            self._eval_vs_pachi()
        else:
            self._eval()


    def _eval_vs_pachi(self):
        agent = self.agent2
        agent_wins = []
        self.env.reset()
        go_state = GoGameState(self.env.board_size



    def _eval(self):
        agent1_wins = []
        for i in range(self.num_games):
            agents = [self.agent1, self.agent2]
            agent_colors = []
            agent_colors[0] = random.choice([1,2])
            agent_colors[1] = 
            cur_player = 0 if agent_colors[0] == 1 else 1

            self.env.reset()
            go_state = GoGameState(self.env.board_size, self.env)
             
            while True:
                if i % 20 == 0:      # print out 5/100 of the games
                    agent_name = self.agent1_name if cur_player == 0 else self.agent2_name
                    print("Agent: " + agent_name)
                    env.render() 

                # Get agent's action to take
                action = agents[cur_player].get_action(go_state)

                # Make that action
                go_state = go_state.succ(action)

                # If there was a winner, finish
                if go_state.winner() != None:
                    agent1_wins.append(go_state.utility(agent_colors[0]))
                    break

                # Update who's turn it is to act
                cur_player = 1 - cur_player

        avg_wins = np.mean(agent1_wins)
        sigma_wins = np.sqrt(np.var(agent1_wins) / len(agent1_wins))

        with open(self.results_filename, "a") as f:
            rslt = self.agent1_name + " vs " self.agent2_name + "\n"
            rslt += "Average wins for agent1: {:04.2f} +/- {:04.2f}".format(avg_wins, sigma_wins) + "\n"
            f.write(rslt)
            print(rslt)











