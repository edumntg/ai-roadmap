# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Loop
        k = 0
        while k < self.iterations:
            V = self.values.copy()

            # Get all states
            states = self.mdp.getStates()
            for state in states:
                # Get all possible actions for this state
                actions = self.mdp.getPossibleActions(state)

                Q_vals = []
                for action in actions:
                    # Compute the Q_start value fot his state and action
                    Q_star = self.computeQValueFromValues(state, action)
                    Q_vals.append(Q_star)

                if len(Q_vals) > 0:
                    V[state] = max(Q_vals) # Take the max Q_star obtained for this state

            self.values = V
            k += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # In this function, we compute the function Q_star
        #
        #   Q_star(s,a) = sum T(s,a,s')*[R(s,a,s') + gamma*V_star(s')]

        # Get all transaction states and their probabilitites
        transitions = self.mdp.getTransitionStatesAndProbs(state, action) # This is a list of all s' and T(s,a,s')

        Q_star= 0.0 # Initial Q value
        for next_state, T in transitions:
            Q_star += T*(self.mdp.getReward(state, action, next_state) + self.discount*self.getValue(next_state))

        return Q_star

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # if self.mdp.isTerminal(state):
        #     return None

        # In this function we return an action (policy). So, in this method
        # we are actually implementing the function pi_start(s)

        #   pi_star(s) = argmax(sum(T(s,a,s')*(R(s,a,s') + gamma*V_start(s'))))
        #   or
        #   pi_star(s) = argmax (Q_star(s,a))

        # First, get all possible actions
        actions = self.mdp.getPossibleActions(state)

        # Initialize a dict for each action
        action_values = util.Counter()

        # Now, loop through each action for each possible action
        for action in actions:
            action_values[action] = self.getQValue(state, action)

        # Finally, return the arg_max value
        return action_values.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

