import numpy as np
import random

# parameters
gamma = 0.9 # discounting rate
rewardSize = -1
gridSize = 4
alpha = 0.5 # (0,1] // stepSize
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 80000

# initialization
V = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# utils
def generateInitialState():
    return random.choice(states[1:-1])

def generateNextAction():
    return random.choice(actions)

def takeAction(state, action):
    if list(state) in terminationStates:
        return 0, None
    finalState = np.array(state)+np.array(action)
    # if robot crosses wall
    if -1 in list(finalState) or gridSize in list(finalState):
        finalState = state
    return rewardSize, list(finalState)       


if __name__ == "__main__":
    print(V)
    for i in range(numIterations):
        state_k = generateInitialState()
        i = 0

        while state_k not in terminationStates:
            i += 1
            action_k = generateNextAction()
            reward,state_k_1 = takeAction(state_k, action_k)

            V_s = V[state_k[0],state_k[1]]
            V_s_1 = V[state_k_1[0],state_k_1[1]]

            V[state_k[0],state_k[1]] = V_s + alpha*(reward+gamma*(V_s_1-V_s))
            
            state_k = state_k_1
            #print(i)
    print(V)