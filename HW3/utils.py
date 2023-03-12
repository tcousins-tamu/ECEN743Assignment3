import gymnasium as gym
import seaborn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from random import choices

#Optimal QVF for plotting purposes only
optQVF = [[0.03928319, 0.03897156, 0.03897156, 0.03182875],
 [0.02111529, 0.02709318, 0.02678154, 0.037495  ],
 [0.05664871, 0.05078747, 0.05676536, 0.03730002],
 [0.02689819, 0.02689819, 0.02103695, 0.03741666],
 [0.06336159, 0.05264813, 0.04550531, 0.02856973],
 [0.,         0.,         0.,         0.        ],
 [0.10128205, 0.08490233, 0.10128205, 0.01637972],
 [0.,         0.,         0.,         0.        ],
 [0.05264813, 0.10210379, 0.08516822, 0.11996007],
 [0.14303455, 0.22793688, 0.19314503, 0.11969418],
 [0.28561382, 0.25574665, 0.21830188, 0.09717911],
 [0.,         0.,         0.,         0.        ],
 [0.,         0.,         0.,         0.        ],
 [0.17555464, 0.29667741, 0.36398935, 0.25574665],
 [0.38157974, 0.63001074, 0.60667037, 0.52647836],
 [0.,         0.,         0.,         0.        ]]
#For answering the questions from the doc:
import csv
outputFile = open("./Results.csv", "w", newline='')
outputWriter = csv.writer(outputFile, delimiter=',')
###############################Learning Section###########################
#contains all code relevant to the "Learning" Section of the hw.
def QLearning(env, gamma=.9, alpha=.5, episodes = 100, epsilon = 0):
    """This function performs the Tabular Q-learning algorithm and returns the policy
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
                    
            #Check if that state has been visited
            #case was added to remove the worthless cases. They were somewhat slowing down convergence
            if nState!=state:
                nVal = QvF[state][action]+alpha*(reward+gamma*(max(QvF[nState]))-QvF[state][action])
                QvF[state][action] = nVal
                state = nState
            
            #if it is a terminating condition, break.
            steps +=1
            if terminal:
                break
        
        #Both alpha and epsilon need to approach 0 (according to notes), but SLOWLY
        alpha*=(episodes-1)/episodes
        #This line will need to be commented out for 5
        epsilon*=(episodes-1)/episodes
        
        #4a, cumulative reward 
        #outputWriter.writerow([episode, np.sum(QvF)])
        
        #4b. Convergence with Q*, also needed for 5
        #outputWriter.writerow([episode, np.sum((QvF-optQVF)**2)**(1/2)])
    #we now need to update the states by their probability of having an outcome happen
    for state in range(numStates):
        nPolicy[state]=np.argmax(QvF[state])
         
    return QvF, nPolicy

def QLearningRand(env, gamma=.9, alpha=.5, episodes = 100):
    """This function performs the Tabular Q-learning algorithm and returns the policy
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
            action = env.action_space.sample()
            #take the action and update the Qvalue function
            nState, reward, terminal, _, _ = env.step(action)
                    
            #Check if that state has been visited
            #case was added to remove the worthless cases. They were somewhat slowing down convergence
            if nState!=state:
                nVal = QvF[state][action]+alpha*(reward+gamma*(max(QvF[nState]))-QvF[state][action])
                QvF[state][action] = nVal
                state = nState
            
            #if it is a terminating condition, break.
            steps +=1
            if terminal:
                break
        
        #Both alpha and epsilon need to approach 0 (according to notes), but SLOWLY
        alpha*=(episodes-1)/episodes
        #This line will need to be commented out for 5
        
        #4a, cumulative reward 
        #outputWriter.writerow([episode, np.sum(QvF)])
        
        #4b. Convergence with Q*, also needed for 5
        #outputWriter.writerow([episode, np.sum((QvF-optQVF)**2)**(1/2)])
    #we now need to update the states by their probability of having an outcome happen
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
                action = choices(actions)[0]
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
        while True:
            if policy is None:
                action = choices(actions)[0]
            else:
                action = policy[state]
            
            #taking a step and calculating the reward
            nState, reward, terminal, _, _ = env.step(action)
            
            vF[state]= vF[state] + alpha*(reward + gamma * vF[nState] - vF[state])
            state = nState

            if terminal:
                break
        alpha*=(episodes-1)/episodes
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
    for iteration in range(iterations):
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
        
        #For Question 1 b
        #outputWriter.writerow([iteration, np.sum((nQvF-QvF)**2)**(1/2)])
        
        err = np.mean((nQvF-QvF)**2)
        QvF = nQvF
        if err < tolerance:
            print("Converged in ", iteration, " iterations")
            break
    #calculating the policy
    for state in range(numStates):
        policy[state]=np.argmax(QvF[state])
    
    return QvF, policy

def PolicyEvaluationSys(numStates, numActions, probMatrix, policy, gamma=.9):
    """Performs policy evaluation through a system of equations

    Args:
        numStates (_type_): _description_
        probMatrix (_type_): _description_
        policy (_type_): _description_
    
    Returns:
        Value Function
    """
    #SEQ FORMULATED AS Vpi = Rpi + PpiVpi
    #Creating the matrix for solving the system using numpy
    def identity(n):
        return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

    probSEQ = np.zeros([numStates, numStates])
    reward = np.zeros([numStates])
    
    for state in range(numStates):
        #these states are terminal
        # if state in [5,7,11,12,15]:
        #     continue
        for action in range(numActions):
            mP = probMatrix[state][action]
            if (policy is None) or (policy[state] == action):
                for movement in mP:
                    prob, nState, _, _ = movement
                    probSEQ[state][nState] += prob

    #We want to remove the terminal state that is 15 from our vF, so
    #we push its const over to 14 (the only state that can access it)
    reward[-2] = probSEQ[14][15] *1
    for idx, row in enumerate(probSEQ):
        sum = np.sum(row)
        if sum!=0:
            probSEQ[idx] = row/sum

    probSEQ = probSEQ*gamma
    vF = np.matmul(np.linalg.inv(np.asarray(identity(numStates)) - probSEQ), reward)
    return vF


def PolicyEvaluation(numStates, probMatrix, policy, gamma=.9, iterations = 100, tolerance = .000005, numActions = None):
    """This function will perform policy evaluation (iterative approach) on a given policy and
    return a corresponding value function.

    Args:
        numStates (int): number of states in the state space
        probMatrix (_type_): transistion matrix, taken from the env
        policy (ndarray): policy to be evaluated. NONE when random policy
        gamma (float, optional):  loss value. Defaults to .9.
        iterations (int, optional): The number of steps taken. Defaults to 100.
        tolerance (float, optional): When MSE reaches this value it will claim convergence.
        numActions (int, optional): ONLY specified when a random policy is being evaluated. This case only occurs
            when policy is also NONE

    Returns:
        ndarray: The value function for the state space
    """
    vF = np.zeros(numStates)
    
    #same deal as QvF with the convergence
    #This case was added in order to allow the uniform random policy outlined in
    #Question 2b
    #print(policy)
    if policy is None:
        actions  = np.asarray(range(numActions))
        
    for iteration in range(iterations):
        nvF = np.zeros(numStates)
        for state in range(numStates):
            if policy is None:
                action = choices(actions)[0]
            else:
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
            #print("Policy Evaluation Converged in ", iteration, "iterations")
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

        #for question 3b
        #outputWriter.writerow([cnt, err])
        
        if err<tolerance:
            print("Policy Iteration Converged in ", cnt, "iterations")
            break
        
    if (cnt == iterations-1):
        print("Did not converge")
            
    return vF, policy

##############Unique Q-Learning Function########################
def QLearningPE(env, gamma=.9, episodes = 100, epsilon = 0, alpha=.9, tolerance=.00005):
    """This function performs the Tabular Q-learning algorithm and returns the policy
    and Q-value function. This one used policy estimation to perform its operations. It does
    this by estimating the policty beforehand. This is not the method done in the slides.

    Args:
        environment (_type_): _description_
        gamma (float, optional): _description_. Defaults to .9.

    Returns:
        _type_: _description_
    """
    def createProbabilityMatrix():
        """This helper function creates a probability matrix by sampling the state space.
        This is done by iterating through the state space at random
        """
        
        fQvF = np.zeros([numStates, numActions])
        for state in range(numStates):
            nProbMatrix[state]={}
            for action in range(numActions):
                nProbMatrix[state][action] = {}
                nProbMatrix[state][action]["numsamples"] = 0
                
        #A probability matrix can be updated while the program is in motion, though this way is easier
        for episode in range(episodes):
            state, _ = env.reset()
            while True:
                #we will make random decisions and figure out the probability of each actions results
                # choice = choices(["max", "random"], [1-epsilon, epsilon])
                # if (choice == ["max"]) and (np.max(fQvF[state]) > 0):
                #     action = np.argmax(fQvF[state])
                # #choose random action
                # else:
                action = env.action_space.sample()
                
                #take the action and update the Qvalue function
                nState, reward, terminal, _, _ = env.step(action)
                
                #Updates the current estimate of the probability matrix
                nSamples =  nProbMatrix[state][action]['numsamples']
                nProbMatrix[state][action]['numsamples'] = nSamples + 1
                if nState not in nProbMatrix[state][action]:
                    nProbMatrix[state][action][nState] = 0
                    
                for outcome in nProbMatrix[state][action]:
                    if outcome == nState:
                        nProbMatrix[state][action][nState]*=nSamples/(nSamples+1)
                        nProbMatrix[state][action][nState]+=1/(nSamples+1)
                    elif outcome!='numsamples':
                        nProbMatrix[state][action][outcome]*=nSamples/(nSamples+1)
                        
                state = nState
                if terminal:
                    break
    
    #Step One: Creating the Q table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    QvF = np.zeros([numStates, numActions])
    
    #This matrix will be setup a bit differently
    #p[state][action]={numSamples}
    nProbMatrix = {}
    createProbabilityMatrix()
               
     #corresponding to the state action pairs
    QvF = np.zeros([numStates, numActions])
    policy = np.zeros(numStates)
    
    #going until the number of iterations has been it, I should 
    for iteration in range(episodes):
        #initialize a new Q-value function to be updated with results from this cycle
        nQvF = np.zeros([numStates, numActions])
        for state in range(numStates):
            nQV = np.zeros(numActions)
            for action in range(numActions):
                moveProb = nProbMatrix[state][action]
                for nState in moveProb:
                    if nState == "numsamples":
                        continue
                    
                    prob = nProbMatrix[state][action][nState]
                    reward = 0
                    if nState == 15:
                        reward = 1
                    #calculating the value of each action and adding it to the new QvF
                    nQV[action]+= prob*(reward+gamma*max(QvF[nState])) #The largest value is the expected decision
            nQvF[state]=nQV
        
        #For Question 1 b;

        
        err = np.mean((nQvF-QvF)**2)
        QvF = nQvF
        #used for demonstrating sum of squaresd
        #outputWriter.writerow([iteration, np.sum((optQVF-QvF)**2)**(1/2)])
        if err < tolerance:
            #print("Converged in ", iteration, " iterations")
            break
    #calculating the policy
    for state in range(numStates):
        policy[state]=np.argmax(QvF[state])
    
    return QvF, policy



#Old
# def PolicyEvaluationSys(numStates, numActions, probMatrix, gamma=.9):
#     """For this implementation, we assume that the policy that we are 
#     checking is optimal, otherwise we could not use the probability matrix.
#     For the suboptimal policy (uniform random) we will use a slightly modified probMatrix
#     that I will be generating.

#     Args:
#         numStates (_type_): _description_
#         probMatrix (_type_): _description_
#         policy (_type_): _description_
#     """
#     #SEQ FORMULATED AS Vpi = Rpi + PpiVpi
#     #Creating the matrix for solving the system using numpy
#     def identity(n):
#         print(n, "N")
#         return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

#     probSEQ = np.zeros([numStates, numStates])
#     const = np.zeros([numStates])
#     print(probSEQ[5])
#     const[-1]=1
#     for state in range(numStates):
#         #these states are terminal
#         if state in [5,7,11,12,15]:
#             continue
#         for action in range(numActions):
#             print(state)
#             mP = probMatrix[state][action]
#             for movement in mP:
#                 prob, nState, _, _ = movement
#                 probSEQ[state][nState] += prob*gamma
    
#     # for idx, row in enumerate(probSEQ):
#     #     if idx in [5,7,11,12,15]:
#     #         continue
#     #     probSEQ[idx] = row/np.sum(row)
#     identity = np.asarray(identity(numStates))
#     # for idx in [5,7,11,12,15]:
#     #     identity[idx][idx]=0
#     probSEQ -= identity
#     #Solutions array
#     np.set_printoptions(threshold=200)
#     np.set_printoptions(linewidth=100)
#     print(probSEQ)
    
#     #const = -1*reward
#     #multiplied by negative 1 because thats how numpy works
#     vF = np.linalg.solve(probSEQ, const)
#     return vF
    