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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions by building a list of scores for each action 
        # as evaluated by the function
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
        #print newPos, newFood, newGhostStates, newScaredTimes 
#==============================================================================
#         print "\nThe New Position: ", newPos
#         print "\nThe Remaining Food: ", newFood
#         print "\nThe New Ghost Positions: ", newGhostStates
#         print "\nThe Number of Scared Moves Left: ", newScaredTimes
#         print "\n-----------------------------------------"
#         
#==============================================================================
        foodlist = currentGameState.getFood().asList()
        #Guard against stopping 
        if action == 'Stop':
            return -float("inf")

        #Guard against PacMan going into a state with a ghost when that ghost is not scared
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos and newScaredTimes == (0,0):
                return -float("inf")
       
        #We want PacMan to move towards the state that is closest to the food!
        foodDistances = []
        for floc in foodlist:
            x = -1 * abs(newPos[0] - floc[0])
            y = -1 * abs(newPos[1] - floc[1])
            foodDistances.append((x+y))
        
        return max(foodDistances)
        
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
        """
        "*** YOUR CODE HERE ***"
        action, numerical = self.value(gameState, 0, 0)
        
        return action 
        
        
    def value (self, gameState, current_index, current_depth):
        #keep track of turns and depth
        if current_index >= gameState.getNumAgents():
            current_index = 0 
            current_depth += 1  
            
        #terminal check     
        if (current_depth == self.depth or gameState.isWin() or gameState.isLose()) :
            return self.evaluationFunction(gameState)
            
        elif current_index == 0 :
            return self.max_value(gameState, current_index, current_depth)
        
        else: 
            return self.min_value(gameState, current_index, current_depth)



    def max_value(self, gameState, current_index, current_depth):    
        #if current_depth == 0:
        v = ("unknown", -float("inf"))
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions:
            return self.evaluationFunction(gameState)
        
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            v_curr = self.value(successor, current_index + 1, current_depth)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
                                
            if v_test > v[1]:
                v = [action, v_test]
        return v 
    
    
    def min_value(self, gameState, current_index, current_depth):
        v = ("unknown", float("inf"))
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions: 
            return self.evaluationFunction(gameState)
                    
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            v_curr = self.value(successor, current_index + 1, current_depth)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
                                
            if v_test < v[1]:
                v = [action, v_test]
        return v
    
    
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, numerical = self.value(gameState, 0, 0, -float("inf"), float("inf"))
        
        return action 
        
        
    def value (self, gameState, current_index, current_depth, alpha, beta):
        #keep track of turns and depth
        if current_index >= gameState.getNumAgents():
            current_index = 0 
            current_depth += 1  
            
        #checks to see if reached terminal states
        if (current_depth == self.depth or gameState.isWin() or gameState.isLose()) :
            return self.evaluationFunction(gameState)
            
        elif current_index == 0 :
            return self.max_value(gameState, current_index, current_depth, alpha, beta)
        
        else: 
            return self.min_value(gameState, current_index, current_depth, alpha, beta)



    def max_value(self, gameState, current_index, current_depth, alpha, beta):    
        #if current_depth == 0:
        v = ("unknown", -float("inf"))
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions:
            return self.evaluationFunction(gameState)
        
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            v_curr = self.value(successor, current_index + 1, current_depth, alpha, beta)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
                                
            if v_test > v[1] :
                v = [action, v_test]
            
            if v_test > beta:
                return v 
            
            alpha = max([alpha, v_test])
        return v 
    
    
    def min_value(self, gameState, current_index, current_depth, alpha, beta):
        v = ("unknown", float("inf"))
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions: 
            return self.evaluationFunction(gameState)
                    
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            v_curr = self.value(successor, current_index + 1, current_depth, alpha, beta)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
                                
            if v_test < v[1]:
                v = [action, v_test]
                
            if v_test < alpha: 
                return v 
            
            beta = min ([beta, v_test])
        return v
    
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
        action, numerical = self.value(gameState, 0, 0)
        
        return action 
        
        
    def value (self, gameState, current_index, current_depth):
        #keep track of turns and depth
        if current_index >= gameState.getNumAgents():
            current_index = 0 
            current_depth += 1  
            
        #terminal check     
        if (current_depth == self.depth or gameState.isWin() or gameState.isLose()) :
            return self.evaluationFunction(gameState)
            
        elif current_index == 0 :
            return self.max_value(gameState, current_index, current_depth)
        
        else: 
            return self.exp_value(gameState, current_index, current_depth)



    def max_value(self, gameState, current_index, current_depth):    
        #if current_depth == 0:
        v = ("unknown", -float("inf"))
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions:
            return self.evaluationFunction(gameState)
        
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            v_curr = self.value(successor, current_index + 1, current_depth)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
                                
            if v_test > v[1]:
                v = [action, v_test]
        return v 
    
    
    def exp_value(self, gameState, current_index, current_depth):
        v = 0
        
        actions = gameState.getLegalActions(current_index)
        
        if not actions: 
            return self.evaluationFunction(gameState)
                    
        for action in actions: 
            successor = gameState.generateSuccessor(current_index, action)
            p = 1/float(len(actions))
            v_curr = self.value(successor, current_index + 1, current_depth)
            
            if type(v_curr) is list: 
                v_test = v_curr[1]
            else:
                v_test = v_curr
            
            v += float(p * v_test)                
                            
        return v
    
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"    

    foodPos = currentGameState.getFood().asList() 
    foodLeft = currentGameState.getNumFood()
    foodDist = [] 
    ghostStates = currentGameState.getGhostStates() 
    capPos = currentGameState.getCapsules()  
    currentPos = list(currentGameState.getPacmanPosition()) 
    
    
    for food in foodPos: 
        foodDist.append(-manhattanDistance(currentPos, food))
    
    if not foodPos:    
        return currentGameState.getScore()
    return currentGameState.getScore() + 1000/float(foodLeft)
    return max(foodDist) + currentGameState.getScore() 
    
# Abbreviation
better = betterEvaluationFunction

