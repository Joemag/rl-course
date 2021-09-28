import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')
S = [(sum_hand, dealer_card, usable_ace) for sum_hand in range(12,22) for dealer_card in range(1,11) for usable_ace in [True, False]]

def monte_carlo_prediction():
    V = {s: 0 for s in S}
    Returns = {s: [] for s in S}

    for i in range(500000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        total_reward = 0
        observed_states = {}
        while not done:
            if obs[0] >= 12 and obs[0] <= 21 and obs not in observed_states:
                # save state and reward up until now
                observed_states[obs] = total_reward
            if obs[0] >= 20:
                #print("stick")
                obs, reward, done, _ = env.step(0)
            else:
                #print("hit")
                obs, reward, done, _ = env.step(1)
            #print("reward:", reward)
            total_reward += reward
        for s in observed_states:
            G = total_reward - observed_states[s]
            Returns[s].append(G) 
            V[s] = np.mean(Returns[s])
            
    X, Y = np.meshgrid(np.arange(12,22),np.arange(1,11))
    Z = np.array([V[(X[x][y], Y[x][y], False)] for x in range(X.shape[0]) for y in range(X.shape[1])])
    Z = Z.reshape(X.shape)
    fig = plt.figure(figsize=(13, 5), dpi=80)
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, color="w", edgecolor="k")
    ax.view_init(40, 210)
    ax.set_xlabel('Player sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('V')
    ax.set_title('No usable ace')
    ax.dist = 13

    Z = np.array([V[(X[x][y], Y[x][y], True)] for x in range(X.shape[0]) for y in range(X.shape[1])])
    Z = Z.reshape(X.shape)
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X, Y, Z, color="w", edgecolor="k")
    ax.view_init(40, 210)
    ax.set_xlabel('Player sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('V')
    ax.set_title('Usable ace')
    ax.dist = 13
    plt.savefig("ex04-mc-prediction.pdf")
    plt.show()


def monte_carlo_es():
    Q = {(s,a): 0 for s in S for a in [0,1]}
    pi = {s: 0 for s in S}
    Returns = {(s,a): [] for s in S for a in [0,1]}
    num_iter = 0
    while True:
        for i in range(100000):
            S0 = S[np.random.choice(len(S))]
            A0 = np.random.choice([0,1])
            #print("S0:", S0)
            obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
            set_state(env, S0)
            obs = env._get_obs()
            #print("observation:", obs)
            next_action = A0
            done = False
            total_reward = 0
            observed_states = {}
            while not done:
                if obs[0] >= 12 and obs[0] <= 21 and (obs,next_action) not in observed_states:
                    # save state and reward up until now
                    observed_states[(obs, next_action)] = total_reward
                obs, reward, done, _ = env.step(next_action)
                
                if obs[0] < 12 or obs[0] > 21:
                    next_action = 1
                else:
                    next_action = pi[obs]
                total_reward += reward


            for (s, a) in observed_states:
                G = total_reward - observed_states[(s,a)]
                Returns[(s,a)].append(G) 
                Q[(s,a)] = np.mean(Returns[(s,a)])
                pi[s] = np.argmax([Q[(s,0)], Q[(s,1)]])
            num_iter += 1
            
        print(f'{num_iter} iterations')
        print('No usable ace')
        print('=============')
        print('     A23456789B')
        print('   #-------------> dealer showing')
        for sum_hand in range(12,22):
            print(f"{sum_hand} | ", end='')
            for dealer_card in range(1,11):
                print(pi[(sum_hand,dealer_card,False)], end='')
            print()
        print("   \\/")
        print("player sum")
        print()
        print('usable ace')
        print('==========')
        print('     A23456789B')
        print('   #-------------> dealer showing')
        for sum_hand in range(12,22):
            print(f"{sum_hand} | ", end='')
            for dealer_card in range(1,11):
                print(pi[(sum_hand,dealer_card,True)], end='')
            print()
        print("   \\/")
        print("player sum")
        print()

def set_state(env, s):
    player_hand = []
    if s[2]:
        while (sum(player_hand) + 11) < s[0]:
            player_hand.append(min(s[0] - (sum(player_hand) + 11), 10))
        player_hand.append(1)
    else:
        while sum(player_hand) < s[0]:
            player_hand.append(min(s[0] - sum(player_hand), 10))
    
    env.player = player_hand
    env.dealer[0] = s[1]


def example_main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    while not done:
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")


def main():
    #monte_carlo_prediction()
    monte_carlo_es()


if __name__ == "__main__":
    main()
