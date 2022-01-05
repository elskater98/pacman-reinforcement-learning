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
    # python autograder.py -q q1
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A counter keeps track of counts for a set of keys. In particular, all keys are defaulted to have value 0.
        self.runValueIteration()

    def runValueIteration(self):
        # Calculate the best action for all states
        for i in range(self.iterations):
            updated_values = util.Counter()

            for state in self.mdp.getStates():
                # Get the best action
                action = self.getAction(state)

                # Check if is not terminal (None)
                if action:
                    # Get the best Q-value for that state
                    updated_values[state] = self.getQValue(state, action)

            self.values = updated_values

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
        q_value = 0

        # Loop for each next state and probabilities; Format: s' and T(s, a, s')
        for nextState, probabilities in self.mdp.getTransitionStatesAndProbs(state, action):
            # Calculate reward for the transition, r(s, a, s')
            reward = self.mdp.getReward(state, action, nextState)

            # Get the V_k(s')
            value = self.getValue(nextState)

            q_value += probabilities * (reward + self.discount * value)

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None

        q_values = util.Counter()

        # Get QValues for each action
        for action in self.mdp.getPossibleActions(state):
            q_values[action] = self.getQValue(state, action)

        # Return the highest QValue
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    # python autograder.py -q q4
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                    [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    # python autograder.py -q q5
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.iterations = iterations
        self.discount = discount
        self.mdp = mdp

        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        old_values = self.values
        predecessors = {}
        priority_queue = util.PriorityQueue()

        for state in states:
            predecessors.update({state: set()})

        for state in states:
            q_values = util.Counter()
            current_value = old_values[state]

            # Loop possible actions
            for action in self.mdp.getPossibleActions(state):

                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

                q_values[action] = self.computeQValueFromValues(state, action)

                # Calculate the difference between q_value and max q_value
                diff = abs(current_value - q_values[q_values.argMax()])

                priority_queue.update(state, -diff)

        for i in range(self.iterations):

            # Check if the priority queue is empty
            if priority_queue.isEmpty():
                break

            state = priority_queue.pop()

            # Calculate best q_value
            if not self.mdp.isTerminal(state):
                best_q_values = util.Counter()

                for action in self.mdp.getPossibleActions(state):
                    best_q_values[action] = self.computeQValueFromValues(state, action)

                old_values[state] = best_q_values[best_q_values.argMax()]

            for p in predecessors[state]:
                q_values_p = util.Counter()

                # Find the best action with the highest QValue
                for action in self.mdp.getPossibleActions(p):
                    q_values_p[action] = self.computeQValueFromValues(p, action)

                diff = abs(old_values[p] - q_values_p[q_values_p.argMax()])

                # Update the priority of this state
                if diff > self.theta:
                    priority_queue.update(p, -diff)
