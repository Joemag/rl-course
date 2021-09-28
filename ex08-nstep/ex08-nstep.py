import gym
import numpy as np
import matplotlib.pyplot as plt


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    def choose_epsilon_greedy_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(env.action_space.n)
        else:
            #return np.argmax(Q[state,:])
            return np.random.choice(np.flatnonzero(Q[state,:] == np.max(Q[state,:])))
    episode_returns = []
    for episode in range(num_ep):
        S = [env.reset()]
        A = [choose_epsilon_greedy_action(S[0])]
        R = [0]
        T = np.inf
        t = 0
        while True:
            if t < T:
                s, r, done, _ = env.step(A[t])
                S.append(s)
                R.append(r)
                if done:
                    T = t + 1
                else:
                    A.append(choose_epsilon_greedy_action(s))
            
            theta = t - n + 1
            if theta >= 0:
                s_theta = S[theta]
                a_theta = A[theta]
                G = 0
                for i in range(theta+1, int(min(theta + n, T)+1)):
                    G += gamma ** (i - theta - 1) * R[i]
                if theta + n < T: 
                    G = G + (gamma**n) * Q[S[theta+n], A[theta+n]]
                Q[s_theta,a_theta] = Q[s_theta,a_theta] + alpha*(G - Q[s_theta,a_theta])

            if theta == T - 1:
                break
            t += 1
        episode_returns.append(sum(R))
    return np.mean(episode_returns)
    # # play
    # for i in range(1):
    #     done = False
    #     s = env.reset()
    #     while not done:
    #         env.render()
    #         a = np.argmax(Q[s,:])
    #         s, r, done, _ = env.step(a)



env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha


for n in [1,2,4,8,16]:#,32,64,128,256,512]:
    alphas = np.linspace(0,1,10)
    performances = []
    for alpha in alphas:
        performances.append(np.mean([nstep_sarsa(env,n,alpha,num_ep=100) for i in range(100)]))
    plt.plot(alphas, performances, label=f"n={n}")
plt.legend()
plt.show()