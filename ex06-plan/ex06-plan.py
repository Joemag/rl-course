import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import cProfile, pstats, io
from pstats import SortKey

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G

longest_path_plot = []
longest_path = 0
def mcts(env, root, maxiter=500):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """

    # this is an example of how to add nodes to the root for all possible actions:
    #root.children = [Node(root, a) for a in range(env.action_space.n)]

    global longest_path_plot
    global longest_path
    for i in range(maxiter):
        # state = copy.deepcopy(env)
        state = gym.make("Taxi-v3")
        state.reset()
        state.env.s = env.env.s
        state.env.lastaction = env.env.lastaction

        G = 0.
        terminal = False
        path_length = 0
        # traverse the tree using an epsilon greedy tree policy
        node = root
        while len(node.children) > 0:
            if np.random.random() < 0.1:
                node = random.choice(node.children)
            else:
                values = [c.sum_value/(c.visits+0.0000001) for c in node.children]  # calculate values for child actions
                node = node.children[np.argmax(values)]  # select the best child
            path_length +=1
            _, reward, terminal, _ = state.step(node.action)
            G += reward
        if path_length > longest_path:
            longest_path = path_length
        longest_path_plot.append(longest_path)
        # Expansion of tree
        if not terminal:
            # add nodes to the root for all possible actions:
            node.children = [Node(node, a) for a in range(env.action_space.n)]
        # This performs a rollout (Simulation):
        if not terminal:
            G += rollout(state)

        # update all visited nodes in the tree
        while node != None:
            node.visits += 1
            node.sum_value += G
            node = node.parent


def main():
    global longest_path
    global longest_path_plot

    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable

    # run the algorithm 10 times:
    rewards = []
    for i in range(10):
        longest_path_plot = []
        mean_return = []
        longest_path = 0
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            env.render()
            mcts(env, root)  # expand tree from root node using mcts
            mean_return.append(root.sum_value/root.visits)
            values = [c.sum_value/c.visits for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action) # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            longest_path -= 1
            sum_reward += reward
        rewards.append(sum_reward)
        print("finished run " + str(i+1) + " with reward: " + str(sum_reward))

        #  plot the mean return over the number of episodes.
        plt.subplot(2, 1, 1)
        plt.plot(range(len(mean_return)), mean_return)
        plt.xlabel("episode")
        plt.ylabel("mean return")
        #  plot the length of the longest path over the number of iterations.
        plt.subplot(2, 1, 2)
        plt.plot(range(len(longest_path_plot)), longest_path_plot)
        plt.xlabel("iteration")
        plt.ylabel("longest path")
        plt.show()
    print("mean reward: ", np.mean(rewards))

if __name__ == "__main__":
    main()
