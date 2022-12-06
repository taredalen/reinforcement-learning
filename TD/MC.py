import numpy as npimport random# variablesgamma = 1rewardSize = -1gridSize = 4terminationStates = [[0, 0], [gridSize - 1, gridSize - 1]]actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]numIterations = 100000# initializationV = np.zeros((gridSize, gridSize))states = [[i, j] for i in range(gridSize) for j in range(gridSize)]# utilsdef generateInitialState():    return random.choice(states[1:-1])def generateNextAction():    return random.choice(actions)def takeAction(state, action):    if list(state) in terminationStates:        return 0, None    finalState = np.array(state) + np.array(action)    # if robot crosses wall    if -1 in list(finalState) or gridSize in list(finalState):        finalState = state    return rewardSize, list(finalState)if __name__ == "__main__":    for i in range(gridSize):        for j in range(gridSize):            # Check we are not in a final state            # if list([i, j]) in terminationStates:            #    continue            reward = 0            current_state = [i, j]            for it in range(numIterations):                state = current_state                while state is not None:                    action = generateNextAction()                    r, state = takeAction(state, action)                    reward += r            V[i, j] = reward / numIterations    print(V)