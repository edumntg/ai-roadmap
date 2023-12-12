# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Expanded set (called closed in pseudocode)
    closed = set()

    # Fringe is a Stack because we are using DFS
    fringe = util.Stack()

    # Insert initial state
    fringe.push((problem.getStartState(), [], 0)) # current state, current moves and current cost

    # Now, loop
    while not fringe.isEmpty():
        # Get current state
        currentState, currentMoves, currentCost = fringe.pop()

        # Check if we are at goal
        if problem.isGoalState(currentState):
            return currentMoves
        
        # If we reach this line, it is because we haven't reached the goal state
        # Check if state is in closed set
        if currentState not in closed:
            closed.add(currentState)
            for state, newMove, cost in problem.getSuccessors(currentState):
                fringe.push((state, currentMoves + [newMove], currentCost + cost))

    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # This one is very similar to DFS but instead of using a Stack, we use a Queue

    # Expanded
    closed = set()

    # Fringe is now a Queue
    fringe = util.Queue()

    # Add initial state, its moves and cost
    fringe.push((problem.getStartState(), [], 0))

    # Now, solve
    while not fringe.isEmpty():
        # Get current state
        currentState, currentMoves, currentCost = fringe.pop()

        # Check if current state is goal state
        if problem.isGoalState(currentState):
            return currentMoves
        
        # Check if current state not in closed set
        if not currentState in closed:
            # Add current state to closed (Visited) set
            closed.add(currentState)

            # Loop through rest
            for state, newMove, cost in problem.getSuccessors(currentState):
                # Add to fringe
                fringe.push((state, currentMoves + [newMove], currentCost + cost))

    return []


    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Expanded (closed) set
    closed = set()

    # Fringe is now a PriorityQueue (the prority is equal to the cost of the state, where lower cost is better)
    fringe = util.PriorityQueue()

    # Add initial state with no moves and zero cost
    fringe.push((problem.getStartState(), [], 0), 0)

    # Loop
    while not fringe.isEmpty():
        # Get current state
        currentState, currentMoves, currentCost = fringe.pop()

        # Check if currentState is goalState
        if problem.isGoalState(currentState):
            return currentMoves
        
        # If current state not in closed set, add it
        if currentState not in closed:
            closed.add(currentState)

            # Loop though rest of states
            for state, newMove, cost in problem.getSuccessors(currentState):
                fringe.push((state, currentMoves + [newMove], currentCost + cost), currentCost + cost)
    
    return []
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Expanded set
    closed = set()

    # We use a priority queue for A*
    fringe = util.PriorityQueue()

    # Add initial state
    fringe.push(
        (problem.getStartState(),
         [],
         (heuristic(problem.getStartState(), problem), 0)
        ), 0) # The cost is now a tuple with heuristic and global costs

    while not fringe.isEmpty():
        # Get current state
        currentState, currentMoves, (hCost, gCost) = fringe.pop()

        # Check if state is goal
        if problem.isGoalState(currentState):
            return currentMoves
        
        # Check if currentState in closed set
        if not currentState in closed:
            closed.add(currentState)

            # Expand rest
            for state, newMove, cost in problem.getSuccessors(currentState):
                # Calculate current heuristic cost
                state_hCost = heuristic(state, problem)
                # Calculate new gCost
                state_gCost = gCost + cost
                # Calculate fCost
                state_fCost = state_hCost + state_gCost
                # Add to fringe
                fringe.push((state, currentMoves + [newMove], (state_hCost, state_gCost)), state_fCost)
    
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
