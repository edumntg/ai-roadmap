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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # print('newFood', newFood.asList())
        # print('newGhostStates', ','.join([str(x) for x in newGhostStates]))
        # print('newScaredTimes', newScaredTimes)
        # print(newFood.count())
        "*** YOUR CODE HERE ***"

        # Compute a score for ghosts
        all_ghosts_cost = []
        for i, ghost in enumerate(newGhostStates):
            ghost_cost = 0 # Default cost for ghost
            # Get ghost position
            ghost_pos = ghost.getPosition()
            # Calculate distance to ghost
            distance_to_ghost = util.manhattanDistance(newPos, ghost_pos)
            # Get scared timer for this ghost
            ghost_timer = newScaredTimes[i]
            # Each movement is a time step, so the ghostScaredTimer is equal to the distance...
            # ...required for pacman to eat it

            # If the timer is higher than the distance, then pacman can reach it
            if ghost_timer > distance_to_ghost:
                ghost_cost = 1000

            # If distance is zero, then pacman is dead
            if distance_to_ghost == 0:
                ghost_cost = -1e9 # pacman died

            all_ghosts_cost.append(ghost_cost)

        # Get distance to closest food
        closest_food_distance = 1e9
        for food_pos in newFood.asList():
            food_distance = manhattanDistance(newPos, food_pos)
            if food_distance < closest_food_distance:
                closest_food_distance = food_distance
        
        # Now, create a score based on food
        foodScore = 1.0/(1.0 + closest_food_distance) # closer is better

        # For ghosts, calculate worst case (min score)
        return successorGameState.getScore() + min(all_ghosts_cost) + foodScore

def scoreEvaluationFunction(currentGameState: GameState):
    
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

    def getAction(self, gameState: GameState):
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
        # Get  all actions
        maxUtility = -1e9
        maxUtilityAction = None
        for action in gameState.getLegalActions(0): # agent 0 is pacman
            successor = gameState.generateSuccessor(0, action)
            stateUtility = self.value(successor, 1, 0)
            if stateUtility > maxUtility:
                maxUtility = stateUtility
                maxUtilityAction = action

        return maxUtilityAction

    def value(self, state: GameState, agentIndex: int, currentDepth: int):
        # Check if state is a terminal state
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        
        # If agent is pacman, maximize value
        if agentIndex == 0: # Pacman is agent 0
            return self.max_value(state, agentIndex, currentDepth)
        else:
            return self.min_value(state, agentIndex, currentDepth)
        
    def max_value(self, state: GameState, agentIndex: int, currentDepth: int):
        # Initialize v
        v = -1e9
        # Now, loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successor, 1, currentDepth))

        return v
    
    def min_value(self, state: GameState, agentIndex: int, currentDepth: int):
        # Initialize v
        v = 1e9
        # Now, loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            if agentIndex == state.getNumAgents() - 1: # Last agent
                v = min(v, self.value(successor, 0, currentDepth + 1)) # Maximize for pacman
            else:
                v = min(v, self.value(successor, agentIndex + 1, currentDepth)) # Minimize for next ghost
    
        return v
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Get successors
        successors = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)] # pacman
        max_utility_action = None
        max_utility = -1e9
        
        # Initialize alpha and beta
        alpha = -1e9
        beta = 1e9

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            state_utility = self.value(successor, 1, 0, alpha, beta)
            if state_utility > max_utility:
                max_utility = state_utility
                max_utility_action = action
                alpha = state_utility

        return max_utility_action

    def value(self, state: GameState, agentIndex: int, currentDepth: int, alpha: float, beta: float):
        # Check if state is terminal state
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        
        if agentIndex == 0:
            return self.max_value(state, agentIndex, currentDepth, alpha, beta)
        else:
            return self.min_value(state, agentIndex, currentDepth, alpha, beta)

    def min_value(self, state: GameState, agentIndex: int, currentDepth: int, alpha: float, beta: float):
        # Initialize v
        v = 1e9
        # Loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            if agentIndex == state.getNumAgents() - 1: # Last agent
                v = min(v, self.value(successor, 0, currentDepth + 1, alpha, beta))
            else:
                v = min(v, self.value(successor, agentIndex + 1, currentDepth, alpha, beta))

            if v < alpha:
                return v
            
            beta = min(beta, v)

        return v

    def max_value(self, state: GameState, agentIndex: int, currentDepth: int, alpha: float, beta: float):
        # Initialize v
        v = -1e9
        # Loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            v = max(v, self.value(successor, 1, currentDepth, alpha, beta))
            if v > beta:
                return v
            
            alpha = max(alpha, v)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_avg_action = None
        max_avg = -1e9
        # Loop through successors
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Compute avg value
            successor_avg = self.value(successor, 1, 0)
            if successor_avg > max_avg:
                max_avg = successor_avg
                max_avg_action = action

        return max_avg_action

    def value(self, state: GameState, agentIndex: int, currentDepth: int):
        # Check ifstate  is a terminal state
        if state.isWin() or state.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(state)
        
        if agentIndex == 0: # pacman
            return self.max_value(state, agentIndex, currentDepth)
        else:
            return self.min_value(state, agentIndex, currentDepth)
            
    def max_value(self, state: GameState, agentIndex: int, currentDepth: int):
        # initialize v
        v = -1e9

        # Loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            # if agent is pacman, to deeper
            v = max(v, self.value(successor, 1, currentDepth))

        return v
    
    def min_value(self, state: GameState, agentIndex: int, currentDepth: int):
        """
            Now instead of the minimum value, we calculate the average
        """
        # Initialize v
        v = 0
        n_states = 0
        # Loop through successors
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            # If agent is last one, go deeper
            if agentIndex == state.getNumAgents() - 1: # last ghost
                v += self.value(successor, 0, currentDepth + 1)
            else: # agent is ghost, examine next ghost
                v += self.value(successor, agentIndex + 1, currentDepth)

        if n_states > 0:
            v = v / n_states

        return v

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    pacmanPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    # Compute a score for ghosts
    all_ghosts_cost = []
    for i, ghost in enumerate(currentGhostStates):
        ghost_cost = 0 # Default cost for ghost
        # Get ghost position
        ghost_pos = ghost.getPosition()
        # Calculate distance to ghost
        distance_to_ghost = util.manhattanDistance(pacmanPos, ghost_pos)
        # Get scared timer for this ghost
        ghost_timer = currentScaredTimes[i]
        # Each movement is a time step, so the ghostScaredTimer is equal to the distance...
        # ...required for pacman to eat it

        # If the timer is higher than the distance, then pacman can reach it
        if ghost_timer > distance_to_ghost:
            ghost_cost = 1000

        # If distance is zero, then pacman is dead
        if distance_to_ghost == 0:
            ghost_cost = -1e9 # pacman died

        all_ghosts_cost.append(ghost_cost)

    # Get distance to closest food
    closest_food_distance = 1e9
    for food_pos in currentFood.asList():
        food_distance = manhattanDistance(pacmanPos, food_pos)
        if food_distance < closest_food_distance:
            closest_food_distance = food_distance
    
    # Now, create a score based on food
    foodScore = 1.0/(1.0 + closest_food_distance) # closer is better

    # For ghosts, calculate worst case (min score)
    return currentGameState.getScore() + min(all_ghosts_cost) + foodScore





# Abbreviation
better = betterEvaluationFunction
