#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import queue

import chess
import numpy as np
import sys

from itertools import permutations


class Aichess():

    """
    A class to represent the game of chess.

    ...

     Attributes:
    -----------
    chess : Chess
        Represents the current state of the chess game.
    listNextStates : list
        A list to store the next possible states in the game.
    listVisitedStates : list
        A list to store visited states during search.
    pathToTarget : list
        A list to store the path to the target state.
    currentStateW : list
        Represents the current state of the white pieces.
    depthMax : int
        The maximum depth to search in the game tree.
    checkMate : bool
        A boolean indicating whether the game is in a checkmate state.
    dictVisitedStates : dict
        A dictionary to keep track of visited states and their depths.
    dictPath : dict
        A dictionary to reconstruct the path during search.

    Methods:
    --------
    getCurrentState() -> list
        Returns the current state for the whites.

    getListNextStatesW(myState) -> list
        Retrieves a list of possible next states for the white pieces.

    isSameState(a, b) -> bool
        Checks whether two states are the same.

    isVisited(mystate) -> bool
        Checks if a state has been visited.

    isCheckMate(mystate) -> bool
        Checks if a state represents a checkmate.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    reconstructPath(state, depth) -> None
        Reconstructs the path to the target state. Updates pathToTarget attribute.

    canviarEstat(start, to) -> None
        Moves a piece from one state to another.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces between states.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    h(state) -> int
        Calculates a heuristic value for a state using Manhattan distance.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    translate(s) -> tuple
        Translates traditional chess coordinates to list indices.

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8
        self.checkMate = False

        # Prepare a dictionary to control the visited state and at which
        # depth they were found
        self.dictVisitedStates = {}
        # Dictionary to reconstruct the BFS path
        self.dictPath = {}


    def getCurrentState(self):
    
        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False


    def isCheckMate(self, mystate):
        
        # list of possible check mate states
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False

    def DepthFirstSearch(self, currentState, depth):
        
        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all noes, we eliminate it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    # in the state son, the first piece is the one just moved
                    # We check the position of currentState
                    # matched by the piece moved
                    if son[0][2] == currentState[0][2]:
                        fitxaMoguda = 0
                    else:
                        fitxaMoguda = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[fitxaMoguda],son[0])
                    # We call again the method with the son, 
                    # increasing depth
                    if self.DepthFirstSearch(son,depth+1):
                        #If the method returns True, this means that there has
                        # been a checkmate
                        # We ad the state to the list pathToTarget
                        self.pathToTarget.insert(0,currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0],currentState[fitxaMoguda])

        # We eliminate the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)

    def worthExploring(self, state, depth):
        
        # First of all, we check that the depth is bigger than depthMax
        if depth > self.depthMax: return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If there state has been visited at a epth bigger than 
                # the current one, we are interestted in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # We update the depth associated to the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # Whenever not visited, we add it to the dictionary 
        # at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True

    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son,depth+1):
                
                # in state 'son', the first piece is the one just moved
                # we check which position of currentstate matche
                # the piece just moved
                if son[0][2] == currentState[0][2]:
                    fitxaMoguda = 0
                else:
                    fitxaMoguda = 1

                # we move the piece to the novel position
                self.chess.moveSim(currentState[fitxaMoguda], son[0])
                # we call the method with the son again, increasing depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns true, this means there was a checkmate
                    # we add the state to the list pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # we return the board to its previous state
                self.chess.moveSim(son[0], currentState[fitxaMoguda])

    def reconstructPath(self, state, depth):
        # When we found the solution, we obtain the path followed to get to this        
        for i in range(depth):
            self.pathToTarget.insert(0,state)
            #Per cada node, mirem quin és el seu pare
            state = self.dictPath[str(state)][0]

        self.pathToTarget.insert(0,state)

    def canviarEstat(self, start, to):
        # We check which piece has been moved from one state to the next
        if start[0] == to[0]:
            fitxaMogudaStart=1
            fitxaMogudaTo = 1
        elif start[0] == to[1]:
            fitxaMogudaStart = 1
            fitxaMogudaTo = 0
        elif start[1] == to[0]:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 1
        else:
            fitxaMogudaStart = 0
            fitxaMogudaTo = 0
        # move the piece changed
        self.chess.moveSim(start[fitxaMogudaStart], to[fitxaMogudaTo])

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next for BFS we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.canviarEstat(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.canviarEstat(moveList[i],moveList[i+1])


    def BreadthFirstSearch(self, currentState, depth):
        """
        Check mate from currentStateW
        """
        BFSQueue = queue.Queue()
        # The node root has no parent, thus we add None, and -1, which would be the depth of the 'parent node'
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there is no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Find the optimal configuration
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If it not the root node, we move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)

            if self.isCheckMate(node):
                # Si és checkmate, construïm el camí que hem trobat més òptim
                self.reconstructPath(node, depthNode)
                break

            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode


    def h(self,state):
        
        if state[0][2] == 2:
            posicioRei = state[1]
            posicioTorre = state[0]
        else:
            posicioRei = state[0]
            posicioTorre = state[1]
        # With the king we wish to reach configuration (2,4), calculate Manhattan distance
        fila = abs(posicioRei[0] - 2)
        columna = abs(posicioRei[1]-4)
        # Pick the minimum for the row and column, this is when the king has to move in diagonal
        # We calculate the difference between row an colum, to calculate the remaining movements
        # which it shoudl go going straight        
        hRei = min(fila, columna) + abs(fila-columna)
        # with the tower we have 3 different cases
        if posicioTorre[0] == 0 and (posicioTorre[1] < 3 or posicioTorre[1] > 5):
            hTorre = 0
        elif posicioTorre[0] != 0 and posicioTorre[1] >= 3 and posicioTorre[1] <= 5:
            hTorre = 2
        else:
            hTorre = 1
        # In our case, the heuristics is the real cost of movements
        return hRei + hTorre

    def AStarSearch(self, currentState):
        '''
        # frontera -> open_list llista de nodes que han set visitats, pero que els seus veïns
        no han set visitats. Comença amb el node start!
        # listVisitedStates : list
        A list to store visited states during search. -> closed_list llista de nodes que han set visitats i els seus veïns també
        open_list = set([start_node])
        closed_list = set([])

        # listVisitedStates es una llista! Representara el estat actual de les nostres fitxes
        '''
        
        frontera = [] # es la meva llista d'opertura 
        frontera.append((self.h(currentState),currentState))

        # conté la distancia del currentState cap els altres nodes
        g = {}
        # currentState es el estat en que ens trobem i l'hem d'inicialitzar a 0
        g[tuple(tuple(row) for row in currentState)] = 0

        #parents contenen un mapa d'adjecencia de tots els nodes
        parents = {}
        parents[tuple(tuple(row) for row in currentState)] = None

        # Mentres la llista frontera no estigui buida, trobem el node amb el menor f(n) al 
        # nostre open list
        while len(frontera) > 0: # frontera not empty si fos cua
           currentState = None
           # aqui recorrem amb un bucle for la nostra llista d'obertura (sfrontera)
           # amb l'objectiu de trobar el node amb el menor cost
           for q in frontera:
               # comprovació amb formula -> f(n) = g(n) + h(n)
            if currentState is None or (g[q] + self.h(q)) < (g[currentState] + self.h(currentState)):
                currentState = q

            if currentState == None:
                   print("path doesn't exists!")
            
            #Si el node que ens trobem es checkMate... es el node objectiu (final)
            objectiu = self.isCheckMate(currentState) # mirem si es true o false

            if objectiu == True:
                #actualitzem la profunditat amb el del currentState
                depth = g[currentState]
                break
        
        # i reconstruim el camí fins al node origen
        self.reconstructPath(currentState, depth) 

        # al tenir la llista dels next stages, podem determinar els parents (familiars) del nostre node
        for neigh in self.getListNextStatesW(currentState):
            if neigh not in self.listVisitedStates:
                if neigh not in frontera:
                    frontera.append(neigh)
                    g[neigh] = g[currentState] + self.h[neigh]
                else:
                    if g[neigh] > g[currentState] + self.h[neigh]:
                        g[neigh] = g[currentState] + self.h[neigh]
        
        frontera.remove(currentState)
        self.listVisitedStates.append(currentState)

	# your code here...



def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None






if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)

    # intiialize board
    TA = np.zeros((8, 8))
    # load initial state
    # white pieces
    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][4] = 12
    
    # initialise bord
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()
    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State",currentState,"\n")

    aichess.AStarSearch(currentState)
    print("#A* move sequence...  ", aichess.pathToTarget)
    print("A* End\n")
    
    print("A* printing end state")
    aichess.chess.boardSim.print_board()



