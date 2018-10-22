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
import numpy as np

from learningAgents import ValueEstimationAgent

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
        self.actions = dict() # keeps track of best actions and Qval for each state
        
        # Write value iteration code here
        "*** YOUR CODE HERE ***"        
        for k in range(self.iterations): #Run value iteration a total of K times
            tempList = [] #Store values so that we can perform batch updates later
            
            for state in self.mdp.getStates(): #Run value iteration on each state 
                stateQVals = []# Store the Q-values for a state
                
                for action in self.mdp.getPossibleActions(state):                    
                    stateQVals += [self.computeQValueFromValues(state,action)]
                                    
                if len(stateQVals) != 0: 
                    tempList += [[state, max(stateQVals) ]] 
                    
            for state in tempList:
                self.values[state[0]] = state[1]
                
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
        #We already know the action so can access reward and transition prob    
        QVal = 0
        for child in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, child[0])
            QVal += child[1] * (reward + self.discount * self.values[child[0]])
        
        return QVal
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None 
        
        QValList = ["", -np.inf]
        for action in self.mdp.getPossibleActions(state): 
            QVal = self.computeQValueFromValues(state, action) 
            if QVal > QValList[1]:
                QValList = [action, QVal]
        return QValList[0]
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
