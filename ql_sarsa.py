### Model free learning using Q-learning and SARSA
### You must not change the arguments and output types of the given function.
### You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from gym.wrappers import Monitor
from taxi_envs import *
from scipy.stats import bernoulli
import random
import matplotlib.pyplot as plt



def QLearning(env, num_episodes, gamma, lr, e):
    """Implement the Q-learning algorithm following the epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int
      Number of episodes of training.
    gamma: float
      Discount factor.
    learning_rate: float
      Learning rate.
    e: float
      Epsilon value used in the epsilon-greedy method.


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    #         YOUR CODE        #

    start_time = time.time()

    # Q initialization
    Q = np.zeros((env.nS, env.nA))


    values= [ ]
    steps_taken= [ ]
    # Episode by episode algorithm iteration and reward calculation
    for eps in range(num_episodes):
        env.reset()
        episode_reward = 0

        s = np.random.choice(env.nS)
        eps_greedy = np.argmax(Q[s, :])

        x = bernoulli.rvs(1 - e)
        if x == 0:
            eps_greedy = np.random.choice(env.nA)

        num_steps = 0
        a = eps_greedy
        terminal = env.P[s][a][0][3]
        while terminal== False:

            eps_greedy = np.argmax(Q[s, :])

            x = bernoulli.rvs(1-e)
            if x == 0:
                eps_greedy = np.random.choice(env.nA)

            a= eps_greedy

            _,sprime,R,terminal= env.P[s][a][0]
            episode_reward+= R

            aprime = np.argmax(Q[sprime, :])

            Q[s][a]+= lr*(R+gamma*Q[sprime][aprime]- Q[s][a])
            s= sprime

            num_steps += 1

        episode_reward = episode_reward/(eps + 1)
        values.append(episode_reward)
        steps_taken.append(num_steps)
        # print('Episode number')
        # print(eps)

    plt.figure( )
    plt.plot(np.arange(num_episodes), values,'k')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative average reward over episodes')
    plt.title('Q Learning')
    plt.grid(True)
    plt.show(block=False)

    plt.figure( )
    plt.plot(np.arange(num_episodes), steps_taken, 'k')
    plt.xlabel('Episodes')
    plt.ylabel('# of steps in each episode')
    plt.title('Q Learning')
    plt.grid(True)
    plt.show(block=False)

    print("--- Q learning execution: %s seconds ---" % (time.time() - start_time))


    return Q


def SARSA(env, num_episodes, gamma, lr, e):
    """Implement the SARSA algorithm following epsilon-greedy exploration.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int
      Number of episodes of training
    gamma: float
      Discount factor.
    learning_rate: float
      Learning rate.
    e: float
      Epsilon value used in the epsilon-greedy method.


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    #         YOUR CODE        #

    start_time = time.time()

    # Q initialization
    Q = np.zeros((env.nS, env.nA))


    values= [ ]
    steps_taken= [ ]

    # Episode by episode algorithm iteration and reward calculation
    for eps in range(num_episodes):
        env.reset()
        episode_reward = 0

        s= np.random.choice(env.nS)
        eps_greedy= np.argmax(Q[s,: ])

        x = bernoulli.rvs(1-e)
        if x==0:
            eps_greedy = np.random.choice(env.nA)


        num_steps= 0
        a= eps_greedy
        terminal= env.P[s][a][0][3]
        while terminal== False:

            _,sprime,R,terminal = env.P[s][a][0]

            eps_greedy = np.argmax(Q[sprime, :])

            x = bernoulli.rvs(1 - e)
            if x == 0:
                eps_greedy = np.random.choice(env.nA)


            aprime= eps_greedy
            episode_reward+= R
            Q[s][a]+= lr*(R+gamma*Q[sprime][aprime]- Q[s][a])
            s= sprime
            a= aprime

            num_steps+= 1


        episode_reward= episode_reward/(eps+1)
        values.append(episode_reward)
        steps_taken.append(num_steps)
        # print('Episode number')
        # print(eps)


    plt.figure( )
    plt.plot(np.arange(num_episodes), values,'r')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative average reward over episodes')
    plt.title('SARSA Learning')
    plt.grid(True)
    plt.show(block=False)

    plt.figure( )
    plt.plot(np.arange(num_episodes), steps_taken, 'r')
    plt.xlabel('Episodes')
    plt.ylabel('# of steps in each episode')
    plt.title('SARSA Learning')
    plt.grid(True)
    print("---Sarsa learning execution: %s seconds ---" % (time.time() - start_time))
    print('Please close the figures if you want to complete program execution')
    plt.show()

    ############################

    return Q


def render_episode_Q(env, Q):
    """Renders one episode for Q function environment.

      Parameters
      ----------
      env: gym.core.Environment
        Environment to play Q function on.
      Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.5)
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print ("Episode reward: %f" %episode_reward)



def main():
    env = gym.make("Assignment1-Taxi-v2")

    Q_QL = QLearning(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    Q_Sarsa = SARSA(env, num_episodes=1000, gamma=0.95, lr=0.1, e=0.1)
    # render_episode_Q(env, Q_QL)

    # Uncomment the following three lines to see output for state-action pair
    # np.set_printoptions(threshold=3000)
    # print(Q_QL)
    # print(Q_Sarsa)


if __name__ == '__main__':
    main()

