import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from copy import deepcopy

#Attempt 2 at a Q Value Function
def QValueIteration(numStates, numActions, probMatrix, gamma=.9, iterations = 100):
    """This function will perform Q-Value iteration on the state space
    and return the policy and values associated with a given state action pair

    Args:
        numStates (int): number of states in the state space
        numActions (int): number of actions in the action space
        probMatrix (_type_): transistion matrix, taken from the env
        gamma (float, optional): loss value. Defaults to .9.
        iterations (int optional): the number of steps taken
        
    Returns:
        tuple: an ndarray containing the Q-value function and the policy
    """

    #corresponding to the state action pairs
    QvF = np.zeros([numStates, numActions])
    policy = np.zeros(numStates)
    
    #going until the number of iterations has been it, I should 
    #add a metric for convergence in the future but this works for now
    for _ in range(iterations):
        #initialize a new Q-value function to be updated with results from this cycle
        nQvF = np.zeros([numStates, numActions])
        for state in range(numStates):
            nQV = np.zeros(numActions)
            for action in range(numActions):
                moveProb = probMatrix[state][action]
                for movement in moveProb:
                    prob, nState, reward, _ = movement
                    #calculating the value of each action and adding it to the new QvF
                    nQV[action]+= prob*(reward+gamma*max(QvF[nState])) #The largest value is the expected decision
            nQvF[state]=nQV
        QvF = nQvF
    
    #calculating the policy
    for state in range(numStates):
        policy[state]=np.argmax(QvF[state])
    
    return QvF, policy

#UPDATE THIS FUNCTION TO DETECT WHEN IT CONVERGES SO WE CAN SAVE TIME IN POLICY ITERATION
def PolicyEvaluation(numStates, probMatrix, policy, gamma=.9, iterations = 100):
    """This function will perform policy evaluation on a given policy and
    return a corresponding value function.

    Args:
        numStates (int): number of states in the state space
        probMatrix (_type_): transistion matrix, taken from the env
        policy (ndarray): policy to be evaluated
        gamma (float, optional):  loss value. Defaults to .9.
        iterations (int, optional): The number of steps taken. Defaults to 100.

    Returns:
        ndarray: The value function for the state space
    """
    vF = np.zeros(numStates)
    
    #same deal as QvF with the convergence
    for _ in range(iterations):
        nvF = np.zeros(numStates)
        for state in range(numStates):
            action = policy[state]
            
            #look at the possible next states
            moveProb = probMatrix[state][action]
            for movement in moveProb:
                prob, nState, reward, _ = movement
                nvF[state] += prob*(reward+gamma*vF[nState])
        vF = nvF
    return vF

def PolicyIteration(numStates, numActions, probMatrix, gamma=.9, iterations = 100):
    """This function will perform policy iteration to arrive at a policy and value function.
    I hesitate to say optimal, becuase that depends on the number of iterations.

    Args:
        numStates (int): number of states in the state space
        numActions (int): number of actions in the state space
        probMatrix (_type_): transisition matrix for the environment
        gamma (float, optional):  loss value. Defaults to .9.
        iterations (int, optional): The number of steps taken. Defaults to 100.

    Returns:
        tuple: tuple containing the value function and policy
    """
    def updatePolicy(numStates, numActions, value, policy, gamma=.9):
        """nested function that serves no purpose outside of its use in policy Iteration
        This function creates an updated policy when given an old one.

        Args:
            numStates (_type_): _description_
            numActions (_type_): _description_
            value (_type_): _description_
            policy (_type_): _description_
            gamma (float, optional): _description_. Defaults to .9.

        Returns:
            _type_: _description_
        """
        return nPolicy
    
    vF = np.zeros(numStates)
    #starting with an "all left" policy
    policy = np.zeros(numStates)
    
    for _ in range(iterations):
            vF = PolicyEvaluation(numStates, numActions, probMatrix, policy, gamma, iterations)
            
    return vF, policy
    
#This function did not work
# def QValueIteration(env, S0, gamma=.9, max_depth=100):
#     """This function will do Q value iteration on the state space
#     and return a Q value function along with a policy

#     Args:
#         A (ndarray): Action Space, corresponds to the possible actions at a space
#         S (ndarray): State Space, corresponds to the possible states
#         S0 (ndarray): Initial State
#         R (ndarray): Reward Function, corresponds to each state
#         gamma (float, optional): _description_. Defaults to .9.
#     Returns;
#         QVF, policy : Tuple of the resulting Q value function and the policy
#     """
#     def QValueRecursion(env, A, S0, policy, QvF, gamma, currentDepth):
#         """Recursive helper function used to arrive at a solution

#         Args:
#             P (ndarray): The current policy corresponding to each state
#             QvF (dict): contains the current mapping of states and actions to values
#         """
#         #If the depth has been maxed out, we call it a day
#         if currentDepth==0:
#             return 0
#         #elsewise we need to decrement it
#         currentDepth = currentDepth-1
        
#         newEnv = deepcopy(env) #have to make a copy of the current state of the environment
#         #Need to take the action at a current state and update the SV;
#         n_state, reward, terminal, truncated, info = newEnv.step(A)
        
#         #print("This is the reward at every step", reward)
#         #if they fall in a hole, theres no more exploration
#         if terminal:
#             return reward
        
#         maxValue = 0
#         for action in range(newEnv.action_space.n):
#             value = reward+gamma*QValueRecursion(newEnv, action, n_state, policy, QvF, gamma, currentDepth)
            
#             #QvF is a dict of dicts
#             #checks if the state action pair can be updated
#             if QvF[S0][A]<value:
#                 QvF[S0][A] = value
            
#             #updates the policy
#             if value > max(QvF[S0].values()):
#                 policy[S0]=action
            
#             #this is separate for cases that are not global maxes
#             if value>maxValue:
#                 maxValue=value

#         if currentDepth%5==0:
#             print("Still Running")
#         return maxValue
            
    
#     policy = [None]*env.observation_space.n #will contain the optimal action for each state
#     QvF = {} #Contains the current maximum value for each state action decision
    
#     #Initializing QvF to have entries for all state action pairs
#     for state in range(env.observation_space.n):
#         QvF[state] = {}
#         for action in range(env.action_space.n):
#             QvF[state][action] = 0
#     #start the recursion at state s0 and go from there
#     for action in range(env.action_space.n):
#         print(max_depth)
#         QValueRecursion(env, action, S0, policy, QvF, gamma, max_depth) 
        
#     return QvF, policy