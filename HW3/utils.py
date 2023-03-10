import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from random import choices
###############################Learning Section###########################
#contains all code relevant to the "Learning" Section of the hw.
#Attempt Two
# epsilon_decay=.6, epsilon_final=.05
def QLearning(env, gamma=.9, episodes = 100, epsilon = 0):
    """This function performs the Q-learning algorithm and returns the policy
    and Q-value function

    Args:
        environment (_type_): _description_
        gamma (float, optional): _description_. Defaults to .9.

    Returns:
        _type_: _description_
    """
    #Step One: Creating the Q table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    QvF = np.zeros([numStates, numActions])
    nPolicy = np.zeros(numStates)
    
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        while True:
            #choose action with the highest value
            choice = choices(["max", "random"], [1-epsilon, epsilon])
            if (choice == ["max"]) and (np.max(QvF[state]) > 0):
                action = np.argmax(QvF[state])
            #choose random action
            else:
                action = env.action_space.sample()
            #take the action and update the Qvalue function
            nState, reward, terminal, _, _ = env.step(action)
            
            #case was added to remove the worthless cases. They were somewhat slowing down convergence
            if nState!=state:
                QvF[state][action] = reward+gamma*(max(QvF[nState]))
                state = nState
            
            #if it is a terminating condition, break.
            steps +=1
            if terminal:
                #on a successful run, we will mitigate the egreedy varaiable
                # epsilon=epsilon*epsilon_decay
                # if epsilon<epsilon_final:
                #     epsilon=epsilon_final
                if nState == 15:
                    print("Converged in: ", episode, " episodes.", "Took: ", steps, " steps")
                break
    
    for state in range(numStates):
        nPolicy[state]=np.argmax(QvF[state])
         
    return QvF, nPolicy

#Monte Carlo Policy Evauluation:
def MonteCarloPolicyEvaluation(env, policy=None, episodes=100, gamma=.9):
    """This will do monte carlo policy evaluation on a given policy.
    This may lead to a diffferent output than anticipated, as it will only
    explore based on the policy provided. NOTE THIS IS NOT EVERY VISIT

    Args:
        env (_type_): environment
        policy (_type_, optional): _description_. Defaults to None.
        episodes (int, optional): _description_. Defaults to 100.
        gamma (float, optional): _description_. Defaults to .9.

    Returns:
        _type_: _description_
    """
    actions  = np.asarray(range(env.action_space.n))
    vF = np.zeros(env.observation_space.n)
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        trajectory = []
        while True:
            if policy is None:
                action = choices(actions)
            else:
                action = policy[state]
            
            #taking a step and calculating the reward
            nState, reward, terminal, _, _ = env.step(action)
            steps+=1
            #updating the trajectory
            trajectory.append(state)
            
            state = nState
            
            if terminal:
                break
        #we know that this works, because the reward of 1 is only at the goal
        for steps, visited in enumerate(reversed(trajectory)):
            vF[visited] += gamma**(steps+1)*reward    
            
    #For our new value function, we want to look to see what the inferred policy is
    vF = vF/episodes
    # for state in range(env.observation_space.n):
        
    return vF

#TD-Learning Implementation
def TDPolicyEvaluation(env, policy=None, episodes=100, gamma=.9, alpha=.5):
    """This will do monte carlo policy evaluation on a given policy.
    This may lead to a diffferent output than anticipated, as it will only
    explore based on the policy provided. NOTE THIS IS NOT EVERY VISIT

    Args:
        env (_type_): environment
        policy (_type_, optional): _description_. Defaults to None.
        episodes (int, optional): _description_. Defaults to 100.
        gamma (float, optional): _description_. Defaults to .9.

    Returns:
        _type_: _description_
    """
    actions  = np.asarray(range(env.action_space.n))
    vF = np.zeros(env.observation_space.n)
    for episode in range(episodes):
        state, _ = env.reset()
        steps = 0
        trajectory = []
        while True:
            if policy is None:
                action = choices(actions)
            else:
                action = policy[state]
            
            #taking a step and calculating the reward
            nState, reward, terminal, _, _ = env.step(action)
            
            vF[state]= vF[state]+ alpha*(reward + gamma * vF[nState] - vF[state])
            state = nState

            if terminal:
                break
            
    #For our new value function, we want to look to see what the inferred policy is
    # vF = vF/episodes
    # for state in range(env.observation_space.n):
    return vF

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