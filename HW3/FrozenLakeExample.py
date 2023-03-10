import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from utils import *

def fancy_visual(value_func, policy_int):
    """This function can be used to generate a heat map that will be displayed. Will show the plot when called.

    Args:
        value_func (ndarray): The value function for this heat map. Each entry corresponds to the value of a square, 1D
        policy_int (ndarray): Corresponding policy to the value function. Corresponds to the action taken at a square, 1D
    """
    grid = 4
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    reshaped = np.reshape(value_func, (grid, grid))
    seaborn.heatmap(reshaped, cmap="icefire", vmax=1.1, robust=True,
                    square=True, xticklabels=grid+1, yticklabels=grid+1,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt="f")
    counter = 0
    for j in range(0, 4):
        for i in range(0, 4):
            if int(policy_int[counter]) == 1:
                plt.text(i+0.5, j+0.7, u'\u2193', fontsize=12)
            elif int(policy_int[counter]) == 3:
                plt.text(i+0.5, j+0.7, u'\u2191', fontsize=12)
            elif int(policy_int[counter]) == 0:
                plt.text(i+0.5, j+0.7, u'\u2190', fontsize=12)
            else:
                plt.text(i+0.5, j+0.7, u'\u2192', fontsize=12)
            counter = counter+1

    plt.title(
        'Heatmap of policy iteration with value function values and directions')
    print('Value Function', value_func)
    print('Policy', policy_int)
    plt.show()


# Setting up the gymnasium environment
#reset the slippery parameter when I finish testing my outputs
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
gamma = 0.9

# Obtaining environment details
# print('Number of Actions', env.action_space.n)
# print('Number of States ', env.observation_space.n)
# print('P[10,3]', env.P[10][3])
# test_value = np.random.rand(16)  # Random Value Function (only for plotting)
# test_policy = np.random.randint(0, 3, 16)  # Random Policy (only for plotting)
# fancy_visual(test_value, test_policy)

#Q Learning example, taken from the test code
# state, info = env.reset()  # Reset the env
# max_step = 20
# for step in range(max_step):
#     action = env.action_space.sample()  # Random Action
#     n_state, reward, terminal, truncated, info = env.step(
#         action)  # Take a step
#     print("Time:", step, 'State:', state, 'Action:', action, 'Reward:',
#           reward, 'Next State:', n_state, 'Terminal:', terminal)
#     state = n_state
#     if terminal or truncated:  # Episode ends if the termination or truncation is true
#         break

##################Begin Assignment Work Here###############
#Q-Value Iteration: implement the Q-Value Function on the frozen lake environment
#a. What is the optimal policy and value function (will need to use the fancy visual function)

state, info = env.reset()
QvF, policy = QValueIteration(env.observation_space.n, env.action_space.n, env.P, gamma=gamma, iterations=100)
#fancy_visual(np.max(QvF, -1), policy)

vF = TDPolicyEvaluation(env, policy, episodes=100, gamma=gamma)
print(vF)
# vF = PolicyEvaluation(env.observation_space.n, env.P, policy, gamma=gamma, iterations=100)
# fancy_visual(vF, policy)

# vF, nP = PolicyIteration(env.observation_space.n, env.action_space.n, env.P, gamma=gamma, iterations=100)
# fancy_visual(vF, nP)
# QvF, policy = QLearning(env, gamma=.9, episodes=5000, epsilon=.5)
# print(QvF)
# fancy_visual(np.max(QvF, -1), policy)