import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

###############################Learning Section###########################
#contains all code relevant to the "Learning" Section of the hw.
def QLearning(environment, gamma=.9):
    return QvF, nPolicy

###############################Planning Section###########################
#Contains all code relevant to the "Planning" Section of the HW
def QValueIteration(numStates, numActions, probMatrix, gamma=.9, iterations = 100, tolerance = .000005):
    """This function will perform Q-Value iteration on the state space
    and return the policy and values associated with a given state action pair

    Args:
        numStates (int): number of states in the state space
        numActions (int): number of actions in the action space
        probMatrix (_type_): transistion matrix, taken from the env
        gamma (float, optional): loss value. Defaults to .9.
        iterations (int optional): the number of steps taken
        tolerance (float, optional): When MSE reaches this value it will claim convergence.
        
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
        
        err = np.mean((nQvF-QvF)**2)
        QvF = nQvF
        if err < tolerance:
            break
    #calculating the policy
    for state in range(numStates):
        policy[state]=np.argmax(QvF[state])
    
    return QvF, policy

def PolicyEvaluation(numStates, probMatrix, policy, gamma=.9, iterations = 100, tolerance = .000005):
    """This function will perform policy evaluation on a given policy and
    return a corresponding value function.

    Args:
        numStates (int): number of states in the state space
        probMatrix (_type_): transistion matrix, taken from the env
        policy (ndarray): policy to be evaluated
        gamma (float, optional):  loss value. Defaults to .9.
        iterations (int, optional): The number of steps taken. Defaults to 100.
        tolerance (float, optional): When MSE reaches this value it will claim convergence.

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

        #checking for convergence and returning
        err = np.mean((nvF-vF)**2)
        vF = nvF
        if err < tolerance:
            break

    return vF

def PolicyIteration(numStates, numActions, probMatrix, gamma=.9, iterations = 100, tolerance = .000005):
    """This function will perform policy iteration to arrive at a policy and value function.
    I hesitate to say optimal, becuase that depends on the number of iterations.

    Args:
        numStates (int): number of states in the state space
        numActions (int): number of actions in the state space
        probMatrix (_type_): transisition matrix for the environment
        gamma (float, optional):  loss value. Defaults to .9.
        iterations (int, optional): The number of steps taken. Defaults to 100.
        tolerance (float, optional): When MSE reaches this value it will claim convergence.

    Returns:
        tuple: tuple containing the value function and policy
    """
    #def updatePolicy(numStates, numActions, probMatrix, value, gamma=.9):
    def updatePolicy():
        """nested function used for policy iteration. Shares same scope so it can access same variables"""
        nPolicy = np.zeros(numStates)
        for state in range(numStates):
            nQ=np.zeros(numActions)
            for action in range(numActions):
                moveProb = probMatrix[state][action]
                for movement in moveProb:
                    prob, nState, reward, _ = movement
                    nQ[action] += prob*(reward+gamma*vF[nState])
            newAction = np.argmax(nQ)
            nPolicy[state] = newAction

        return nPolicy
    
    vF = np.zeros(numStates)
    #starting with an "all left" policy
    policy = np.zeros(numStates)
    
    cnt = 0
    for cnt in range(iterations):
        vF = PolicyEvaluation(numStates, probMatrix, policy, gamma, iterations, tolerance)
        
        nPolicy = updatePolicy() #updating the policy based on new value function. 

        err = np.linalg.norm(nPolicy-policy)
        policy = nPolicy

        if err==0:
            break
        
    if (cnt == iterations-1):
        print("Did not converge")
            
    return vF, policy