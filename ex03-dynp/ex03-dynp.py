import gym
import numpy as np
import math as math

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    nsteps = 0
    while True:
        delta = 0
        for s in range(n_states):
            v = V_states[s]
            V_states[s] = max([math.fsum([p * (r + gamma * V_states[n_state]) for p, n_state, r, is_terminal in env.P[s][a]]) for a in range(n_actions)])
            delta = max(delta, abs(v - V_states[s]))
        nsteps += 1
        if delta < theta:
            break
    print("Number of steps: {}".format(nsteps))
    return V_states

def make_policy(value):
    policy = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8

    for s in range(n_states):
        policy[s] = np.argmax([math.fsum([p * (r + gamma * value[n_state]) for p, n_state, r, is_terminal in env.P[s][a]]) for a in range(n_actions)])

    return policy

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    value = value_iteration()
    print("Computed value:")
    print(value)
    policy = make_policy(value)
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
