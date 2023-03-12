import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from utils import *

np.set_printoptions(threshold=200)
np.set_printoptions(linewidth=100)

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
    # print('Value Function', value_func)
    # print('Policy', policy_int)
    plt.show()


# Setting up the gymnasium environment
#reset the slippery parameter when I finish testing my outputs
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
gamma = 0.9
alpha = .1

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

#Creating the optimal value function through QVI
rewards = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0])
################Question 1###############################################
print("#####Question One#################################################")
################1a. What is the optimal policy and value function?#######
print("1a. What is the optimal Policy and value function?")
state, info = env.reset()
optQvF, optPolicy = QValueIteration(env.observation_space.n, env.action_space.n, env.P, gamma=gamma, iterations=100)
print("Optimal policy: \n", optPolicy)
print("\nOptimal Value Function:\n", optQvF)

#########1b. Plot Uk (used the commented out lines to write to excel)###
#This section has no code, in order to do this, uncomment the csv file information 
#at top of utils and uncomment the write sction in QValueIteration

###############1c. Plot the heat map##################################### 
#fancy_visual(np.max(optQvF, -1), optPolicy)

# vF = TDPolicyEvaluation(env, policy, episodes=100, gamma=gamma)
# print(vF)

################Question 2###############################################
print("\n#####Question Two#################################################")
################2a. Perform policy evaluation using a systemn of eq######
print("2a. Obtain Value Function from a system of equations")
#vF = PolicyEvaluation(env.observation_space.n, env.P, optPolicy, gamma=gamma, iterations=100000)
vF = PolicyEvaluationSys(env.observation_space.n, env.action_space.n, env.P, optPolicy, gamma=gamma)
#fancy_visual(vF, optPolicy)
print("Value Function obtained by system of EQ (optimal Policy):\n", vF)

vF = PolicyEvaluationSys(env.observation_space.n, env.action_space.n, env.P, None, gamma=gamma)
#fancy_visual(vF, optPolicy)
print("\nValue Function obtained by system of EQ (random Policy):\n", vF)

################2b. Perform policy evaluation using a systemn of eq#######
print("2b. Obtain Value Function from the iterative approach")
state, info = env.reset()
vF = PolicyEvaluation(env.observation_space.n, env.P, optPolicy, gamma=gamma, iterations=100)
#fancy_visual(vF, optPolicy)
print("Value Function obtained through iterative approach on optimal policy:\n", vF)

#If I dont specify a policy it uses uniform random
state, info = env.reset()
vF = PolicyEvaluation(env.observation_space.n, env.P, None, gamma=gamma, iterations=1000, numActions=env.action_space.n)
#fancy_visual(vF, optPolicy)
print("\nValue Function obtained through iterative approach on random policy:\n", vF)
################2c. which is better?######################################
#This is a purely written response

##############Question 3##################################################
print("\n#####Question Three################################################")
#######3a. policy iteration to get optimal policy and value function######
print("3a. Optimal Policy and value function from Policy Iteration")
state, info = env.reset()
vF, nP = PolicyIteration(env.observation_space.n, env.action_space.n, env.P, gamma=gamma, iterations=100)
# fancy_visual(vF, nP)
print("Optimal Value Function: \n", vF)
print("\nOptimal Policy: \n", nP)

#######3b. Compare the  convergence of QVI and PI#######################
#This section has no code, in order to do this, uncomment the csv file information 
#at top of utils and uncomment the write sction in PolicyITeration

##############Question 4##################################################
print("\n#####Question Four################################################")
#######4a. QLearning######################################################
#In order to get cumulative reward, you will need to uncomment the 4a
#section in QLearning as well as the csv file at the top of utils
print("4a. Optimal Policy and value function from Policy Iteration")
state, info = env.reset()
QvF, nPolicy = QLearning(env, gamma=gamma, alpha=alpha, episodes = 10000, epsilon=.9)
print("Optimal Q Value Function Under Q-Learning: \n", QvF)
print("\nOptimal Policy under Q-Learning: \n", nPolicy)
#fancy_visual(np.max(QvF, -1), nPolicy)

#######4b. QLearning Conv#################################################
#For 4B, you will need to follow similar instructions. You will need to 
#uncomment the csv thing at the top of utils as well as uncommment the 4b
#section. This will be a part of running Q-Learning, so an additional run here
#is unnecessary

#######4c. QLearning Output###############################################
#This is given from 4A

##############Question 5##################################################
#In order to get graph, will need to uncomment lines in function as always
print("\n#####Question Five################################################")
print("5. Uniform Random Policy")
state, info = env.reset()
QvF, nPolicy = QLearningRand(env, gamma=gamma, alpha=alpha, episodes = 10000)
print("Optimal Q Value Function Under Q-Learning w/ Random Policy: \n", QvF)
print("\nOptimal Policy under Q-Learning w/ Random Policy: \n", nPolicy)
#fancy_visual(np.max(QvF, -1), nPolicy)

#This function differs from traditional Q-Learning in that it traverses the state space
#In order to guess the policy, the does QVI on this estimated policy
print("\n5. Policy Estimation Q-Learning\n")
state, info = env.reset()
QvF, nPolicy = QLearningPE(env, gamma=gamma, episodes=10000, tolerance=.0000000005)
print("Q Learning PE QVF: \n", QvF)
print("\nQ Learning PE Policy: \n", nPolicy)
#fancy_visual(np.max(QvF, -1), nPolicy)
#vF = PolicyEvaluation(env.observation_space.n, env.P, policy, gamma=gamma, iterations=1000)


#Question 6############################################################
print("\n#####Question Six################################################")
#######6a. MonteCarlo######################################################
print("6a. Monte Carlo Policy Evaluation")
#Optimal Policy
state, info = env.reset()
vF = MonteCarloPolicyEvaluation(env, optPolicy, episodes=1000, gamma = gamma)
print("Value function from MC on optimal Policy: \n", vF)
#fancy_visual(vF, optPolicy)

#Random Policy
state, info = env.reset()
vF = MonteCarloPolicyEvaluation(env, None, episodes=1000, gamma = gamma)
#fancy_visual(vF, optPolicy)
print("\nValue function from MC on Random Policy: \n", vF)

print("\n6b. TD Learning")
#Optimal Policy
state, info = env.reset()
vF = TDPolicyEvaluation(env, optPolicy, episodes=1000, gamma = gamma, alpha=alpha)
#fancy_visual(vF, optPolicy)
print("Value function from TD on optimal Policy: \n", vF)

#Random Policy
state, info = env.reset()
state, info = env.reset()
vF = TDPolicyEvaluation(env, None, episodes=1000, gamma = gamma, alpha=alpha)
#fancy_visual(vF, optPolicy)
print("\nValue function from TD on random Policy: \n", vF)

#This line is added to close the file that I open as a global in my utils for answering questions in the hw
outputFile.close()