import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

aggregation_size = 20
def plot_V(Q):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    dims = (aggregation_size, aggregation_size)
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
    plt.imshow(V, origin='upper', 
               extent=[0,dims[0],0,dims[1]], vmin=np.min(Q), vmax=np.max(Q), 
               cmap=plt.cm.RdYlGn, interpolation='none')
    # for x, y in product(range(dims[0]), range(dims[1])):
        # plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(V[x,y]),
            #    horizontalalignment='center', 
            #    verticalalignment='center')
    plt.xticks(range(aggregation_size+1))
    plt.yticks(range(aggregation_size+1))
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.show()

def map_range(start1, end1, start2, end2, value):
    return ((value - start1) / (end1 - start1)) * (end2 - start2) + start2




def aggregate_state(s):
    pos = int(np.floor(map_range(-1.2, 0.6, 0, aggregation_size, s[0])))
    velocity = int(np.floor(map_range(-0.07, 0.07, 0, aggregation_size, s[1])))
    return np.ravel_multi_index((velocity, pos), (aggregation_size, aggregation_size))

def play_greedy(env, Q):
    def choose_greedy_action(state):
        return np.argmax(Q[state,:])
    s = aggregate_state(env.reset())
    steps = 0
    while True:
        env.render()
        a = choose_greedy_action(s)
        observation, r, done, _ = env.step(a)
        steps += 1
        s_ = aggregate_state(observation)
        s=s_
        if done:
            break
    return steps, observation[0] >= 0.5

def qlearning_episode(env, Q, i, alpha=0.1, gamma=0.9, epsilon=0.1):
    def choose_greedy_action(state):
        return np.argmax(Q[state,:])
    def choose_epsilon_greedy_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.action_space.n)
        else:
            return np.argmax(Q[state,:])
            return np.random.choice(np.flatnonzero(Q[state,:] == np.max(Q[state,:])))
    s = aggregate_state(env.reset())
    steps = 0
    while True:
        #if(i>1600):
            #print(np.array2string(Q,edgeitems = 400))
            #env.render()
        
        a = choose_epsilon_greedy_action(s)
        #print("do action: ", a)
        observation, r, done, _ = env.step(a)
        steps += 1
        s_ = aggregate_state(observation)
        Q[s,a] = Q[s,a] + alpha*(r + (gamma * Q[s_,choose_greedy_action(s_)]) - Q[s,a])
        
        s=s_
        #print("observation: ", observation)
        #print("reward: ", r)
        #print("")
        if done:
            break
    return steps, observation[0] >= 0.5

def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=2000):
    Q = np.zeros((aggregation_size*aggregation_size,  3))

    num_steps = []
    cumulative_success = []
    current_cumulative_success = 0
    for j in range(int(num_ep/20)):
        for i in range(20):
            steps, success = qlearning_episode(env, Q, i, alpha, gamma, epsilon)
            num_steps.append(steps)
            current_cumulative_success += float(success)
            cumulative_success.append(current_cumulative_success)
        #plot_V(Q)
    while True:
        play_greedy(env, Q)
    return num_steps, cumulative_success
def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break


def main():
    env = gym.make('MountainCar-v0')
    env.reset()

    all_num_steps, all_cumulative_success = [], []
    for i in range(10):
        num_steps, cumulative_success = qlearning(env)
        all_num_steps.append(num_steps)
        all_cumulative_success.append(cumulative_success)

    plt.plot(range(len(all_num_steps[0])), np.mean(np.array(all_num_steps), axis = 0))
    plt.xlabel("episode")
    plt.ylabel("avg steps")
    plt.show()

    plt.plot(range(len(all_cumulative_success[0])), np.mean(np.array(all_cumulative_success), axis = 0))
    plt.xlabel("episode")
    plt.ylabel("avg cumulative success")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
