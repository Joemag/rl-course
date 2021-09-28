import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    e0 = np.exp(np.dot(theta[:,0], state)) 
    e1 = np.exp(np.dot(theta[:,1], state))
    return [e0/(e0+e1), e1/(e0+e1)]


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    gamma = 0.99
    alpha = 0.02
    theta = np.random.rand(4, 2)  # policy parameters

    last_hundred_episode_lengths = []
    episode_length_means = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean
        last_hundred_episode_lengths.append(len(states))
        if len(last_hundred_episode_lengths) > 100:
            last_hundred_episode_lengths = last_hundred_episode_lengths[1:]
        length_mean = np.mean(last_hundred_episode_lengths)
        episode_length_means.append(length_mean)
        #if length_mean == 500: return episode_length_means
        print("Mean: "+ str(length_mean))

        # TODO: implement the reinforce algorithm to improve the policy weights
        T = len(states)
        for t in range(T):
            Gt = np.sum([gamma**(k-t) * rewards[k] for k in range(t, T)])
            grad_log = np.multiply((1 - policy(states[t], theta)[actions[t]]), states[t])
            theta[:,actions[t]] = theta[:,actions[t]] + alpha*(gamma**t) * Gt * grad_log
    return episode_length_means

def REINFORCE_with_baseline(env):
    gamma = 0.99
    alpha_theta = 0.02
    alpha_w = 0.02
    theta = np.random.rand(4, 2)  # policy parameters
    weights = np.zeros(4)

    last_hundred_episode_lengths = []
    episode_length_means = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean
        last_hundred_episode_lengths.append(len(states))
        if len(last_hundred_episode_lengths) > 100:
            last_hundred_episode_lengths = last_hundred_episode_lengths[1:]
        length_mean = np.mean(last_hundred_episode_lengths)
        episode_length_means.append(length_mean)
        #if length_mean == 500: return episode_length_means
        print("Mean: "+ str(length_mean))

        # TODO: implement the reinforce algorithm to improve the policy weights
        T = len(states)
        for t in range(T):
            Gt = np.sum([gamma**(k-t) * rewards[k] for k in range(t, T)])
            delta = Gt - np.dot(weights, states[t])
            weights = weights + alpha_w * (gamma**t) * delta * states[t]
            grad_log = np.multiply((1 - policy(states[t], theta)[actions[t]]), states[t])
            theta[:,actions[t]] = theta[:,actions[t]] + alpha_theta*(gamma**t) * delta * grad_log
    return episode_length_means


def main():
    env = gym.make('CartPole-v1')
    reinforce_lengths = REINFORCE(env)
    reinforce_baseline_lengths = REINFORCE_with_baseline(env)
    plt.plot(range(len(reinforce_lengths)),reinforce_lengths, label="REINFORCE")
    plt.plot(range(len(reinforce_baseline_lengths)),reinforce_baseline_lengths, label="REINFORCE with baseline")
    plt.xlabel("episode")
    plt.ylabel("episode length")
    plt.legend()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
