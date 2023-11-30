# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        inf = float('inf')
        # calculate the distance from each ghost to pacman current position
        for ghost in newGhostStates:
            ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition())]

        # calculate ghost score to consider the danger of the nearest ghost
        for distance, scared in zip(ghostDistances, newScaredTimes):
            ghostScore = min([-distance if distance <= 1 and scared <= distance else 0])

        # calculate the distance from pacman to the closest food
        food_distances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        distToClosestFood = min(food_distances, default=inf)

        # increase a score to make pacman move closer to remaining closest food
        closestFoodScore = 1.0 / (1.0 + distToClosestFood)

        finalScore = successorGameState.getScore() + closestFoodScore + ghostScore
        return finalScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            bestValue = float("-inf")
            bestAction = None
            for action in state.getLegalActions(0):  # Pacman is agent 0
                successorState = state.generateSuccessor(0, action)
                value, _ = minValue(successorState, depth, 1)  # Start with the first ghost
                if value > bestValue:               # update best value and action if it's the better outcome
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction

        def minValue(state, depth, ghostIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            bestValue = float("inf")
            bestAction = None
            for action in state.getLegalActions(ghostIndex):
                successorState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:  # check for the last ghost
                    value, _ = maxValue(successorState, depth - 1)
                else:                                       # iterate through the remaining ghosts
                    value, _ = minValue(successorState, depth, ghostIndex + 1)
                if value < bestValue:               # update best value and action if it's the better outcome
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction

        _, bestAction = maxValue(gameState, self.depth)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            bestValue = float("-inf")
            bestAction = None
            for action in state.getLegalActions(0):  # Pacman is agent 0
                successorState = state.generateSuccessor(0, action)
                value, _ = minValue(successorState, depth, 1, alpha, beta)  # Start with the first ghost

                # update best value and action if it's the better outcome
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                if bestValue > beta:    # prune the search tree if necessary
                    return bestValue, bestAction
                alpha = max(alpha, bestValue)   # update alpha
            return bestValue, bestAction

        def minValue(state, depth, ghostIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            bestValue = float("inf")
            bestAction = None
            for action in state.getLegalActions(ghostIndex):
                successorState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:  # check for the last ghost
                    value, _ = maxValue(successorState, depth - 1, alpha, beta)
                else:  # iterate through the remaining ghosts
                    value, _ = minValue(successorState, depth, ghostIndex + 1, alpha, beta)

                # update best value and action if it's the better outcome
                if value < bestValue:
                    bestValue = value
                    bestAction = action
                if bestValue < alpha:    # prune the search tree if necessary
                    return bestValue, bestAction
                beta = min(beta, bestValue)     # update beta
            return bestValue, bestAction

        _, bestAction = maxValue(gameState, self.depth, float("-inf"), float("inf"))
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # recursive call the expctimax search
        _, action = self.expectimax(gameState, self.depth, 0)
        return action

    def expectimax(self, state, depth, ghostIndex):
        if  state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), None

        # pacman(max)
        if ghostIndex == 0:
            bestValue = float("-inf")
            bestAction = None
            for action in state.getLegalActions(ghostIndex):
                successorState = state.generateSuccessor(ghostIndex, action)
                value, _ = self.expectimax(successorState, depth, ghostIndex + 1)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return bestValue, bestAction

        # ghosts(average)
        else:
            totalValue = 0
            numActions = len(state.getLegalActions(ghostIndex))
            for action in state.getLegalActions(ghostIndex):
                successorState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == state.getNumAgents() - 1:  # check for the last ghost
                    value, _ = self.expectimax(successorState, depth - 1, 0)
                else:                       # iterate through the remaining ghosts
                    value, _ = self.expectimax(successorState, depth, ghostIndex + 1)
                totalValue += value
            return totalValue / numActions, None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # the current game score
    score = currentGameState.getScore()

    # assume some constants for score calculation
    inf = float("inf")
    foodScore = 10.0                # food score to encourage pacman to eat
    ghostScore = -10.0              # ghost score to discourage pacman to come close
    scaredGhostScore = 100.0        # scared ghost score to encourage pacman to eat

    # calculate the distance from pacman to the closest food
    foodDistances = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    # if there's food left
    if len(foodDistances) > 0:
        score += foodScore / min(foodDistances)
    # if there's no food left
    else:
        score += foodScore

    # calculate the distance from each ghost to pacman current position
    for ghost in newGhostStates:
        ghostDistances = util.manhattanDistance(newPos, ghost.getPosition())
        if ghostDistances > 0:
            # if ghosts are scared
            if ghost.scaredTimer > 0:
                score += scaredGhostScore / ghostDistances
            # if ghosts are not scared
            else:
                score += ghostScore / ghostDistances
        # if pacman got caught by a ghost
        else:
            return -inf

    return score

# Abbreviation
better = betterEvaluationFunction
