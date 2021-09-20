from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        self.validSet=(0,1,2)
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]
        
        #Start local board index for reflex agent playing
        self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')

    def twoInARow(self, player1, player2):
        '''
        This is a helper function to count the number of two-in-a-row:
            (1) without the third spot taken by the opposing player
            (2) that you've prevented the opponent player forming two-in-a-row
        input args:
        player1: the current player, either 'X' (maxPlayer) or 'O' (minPlayer)
        player2: the opponent player, either 'X' (maxPlayer), 'O' (minPlayer), or '_'
        output:
        twos: the number of two-in-a-row
        '''
        twos = 0 
        # Count twos in a row, denote "1" as player1, "0" as player2
        for index in self.globalIdx:
            # First row, 110, 101, 011
            if (self.board[index[0]][index[1]] == player1 and self.board[index[0]][index[1]+1] == player1 and self.board[index[0]][index[1]+2] == player2) or (self.board[index[0]][index[1]] == player1 and self.board[index[0]][index[1]+1] == player2 and self.board[index[0]][index[1]+2] == player1) or (self.board[index[0]][index[1]] == player2 and self.board[index[0]][index[1]+1] == player1 and self.board[index[0]][index[1]+2] == player1): twos +=1
            # Second row, 110, 101, 011
            if (self.board[index[0]+1][index[1]] == player1 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+1][index[1]+2] == player2) or (self.board[index[0]+1][index[1]] == player1 and self.board[index[0]+1][index[1]+1] == player2 and self.board[index[0]+1][index[1]+2] == player1) or (self.board[index[0]+1][index[1]] == player2 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+1][index[1]+2] == player1): twos +=1
            # Third row, 110, 101, 011
            if (self.board[index[0]+2][index[1]] == player1 and self.board[index[0]+2][index[1]+1] == player1 and self.board[index[0]+2][index[1]+2] == player2) or (self.board[index[0]+2][index[1]] == player1 and self.board[index[0]+2][index[1]+1] == player2 and self.board[index[0]+2][index[1]+2] == player1) or (self.board[index[0]+2][index[1]] == player2 and self.board[index[0]+2][index[1]+1] == player1 and self.board[index[0]+2][index[1]+2] == player1): twos +=1
            # First column, 110, 101, 011
            if (self.board[index[0]][index[1]] == player1 and self.board[index[0]+1][index[1]] == player1 and self.board[index[0]+2][index[1]] == player2) or (self.board[index[0]][index[1]] == player1 and self.board[index[0]+1][index[1]] == player2 and self.board[index[0]+2][index[1]] == player1) or (self.board[index[0]][index[1]] == player2 and self.board[index[0]+1][index[1]] == player1 and self.board[index[0]+2][index[1]] == player1): twos +=1
            # Second column, 110, 101, 011
            if (self.board[index[0]][index[1]+1] == player1 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]+1] == player2) or (self.board[index[0]][index[1]+1] == player1 and self.board[index[0]+1][index[1]+1] == player2 and self.board[index[0]+2][index[1]+1] == player1) or (self.board[index[0]][index[1]+1] == player2 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]+1] == player1): twos +=1
            # Third column, 110, 101, 011
            if (self.board[index[0]][index[1]+2] == player1 and self.board[index[0]+1][index[1]+2] == player1 and self.board[index[0]+2][index[1]+2] == player2) or (self.board[index[0]][index[1]+2] == player1 and self.board[index[0]+1][index[1]+2] == player2 and self.board[index[0]+2][index[1]+2] == player1) or (self.board[index[0]][index[1]+2] == player2 and self.board[index[0]+1][index[1]+2] == player1 and self.board[index[0]+2][index[1]+2] == player1): twos +=1
            # Diagonals
            if (self.board[index[0]][index[1]] == player1 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]+2] == player2) or (self.board[index[0]][index[1]] == player1 and self.board[index[0]+1][index[1]+1] == player2 and self.board[index[0]+2][index[1]+2] == player1) or (self.board[index[0]][index[1]] == player2 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]+2] == player1): twos +=1 
            if (self.board[index[0]][index[1]+2] == player1 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]] == player2) or (self.board[index[0]][index[1]+2] == player1 and self.board[index[0]+1][index[1]+1] == player2 and self.board[index[0]+2][index[1]] == player1) or (self.board[index[0]][index[1]+2] == player2 and self.board[index[0]+1][index[1]+1] == player1 and self.board[index[0]+2][index[1]] == player1): twos +=1 
        
        return twos

    def threeInARow(self, player):
        '''
        This is a helper function to check if there exists three-in-a-row.
        input args:
        player: the current player, either 'X' (maxPlayer) or 'O' (minPlayer)
        output:
        three: '1' if there exists three-in-a-row, '0' otherwise
        '''
        for index in self.globalIdx:
            # Check rows
            if self.board[index[0]][index[1]] == player and self.board[index[0]][index[1]+1] == player and self.board[index[0]][index[1]+2] == player: return 1
            if self.board[index[0]+1][index[1]] == player and self.board[index[0]+1][index[1]+1] == player and self.board[index[0]+1][index[1]+2] == player: return 1
            if self.board[index[0]+2][index[1]] == player and self.board[index[0]+2][index[1]+1] == player and self.board[index[0]+2][index[1]+2] == player: return 1
            # Check columns
            if self.board[index[0]][index[1]] == player and self.board[index[0]+1][index[1]] == player and self.board[index[0]+2][index[1]] == player: return 1
            if self.board[index[0]][index[1]] == player and self.board[index[0]+1][index[1]+1] == player and self.board[index[0]+2][index[1]+1] == player: return 1
            if self.board[index[0]][index[1]] == player and self.board[index[0]+1][index[1]+2] == player and self.board[index[0]+2][index[1]+2] == player: return 1
            # Check diagonals
            if self.board[index[0]][index[1]] == player and self.board[index[0]+1][index[1]+1] == player and self.board[index[0]+2][index[1]+2] == player: return 1
            if self.board[index[0]+2][index[1]] == player and self.board[index[0]+1][index[1]+1] == player and self.board[index[0]][index[1]+2] == player: return 1

        return 0

    def countCorners(self, player):
        '''
        This is a helper function to count the number of corners taken by the offensive/defensive player.
        input args:
        player: the current player, either 'X' (maxPlayer) or 'O' (minPlayer)
        output:
        corners: the number of corners taken by the offensive/defensive player
        '''
        corners = 0
        for index in self.globalIdx:
            if self.board[index[0]][index[1]] == player: corners += 1
            if self.board[index[0]+2][index[1]] == player: corners += 1
            if self.board[index[0]][index[1]+2] == player: corners += 1
            if self.board[index[0]+2][index[1]+2] == player: corners += 1
        return corners

    def possibleMoves(self, currBoardIdx):
        '''
        This function checks possible move options in a local board.
        input args:
        currBoardIdx: current local board index
        output:
        possibleMoves: possible move options within the current board
        '''
        boardIdx = self.globalIdx[currBoardIdx]
        possibleMoveOptions = []
        for i in range(3):
            for j in range(3):
                if (self.board[boardIdx[0]+i][boardIdx[1]+j] != self.maxPlayer) and (self.board[boardIdx[0]+i][boardIdx[1]+j] != self.minPlayer):
                    possibleMoveOptions.append((boardIdx[0]+i, boardIdx[1]+j))
        return possibleMoveOptions

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0

        if isMax:
            # First rule
            if self.checkWinner() == 1: return 10000
            
            # Second rule
            score += 500 * self.twoInARow(self.maxPlayer, '_')
            score += 100 * self.twoInARow(self.maxPlayer, self.minPlayer)

            if score != 0: return score

            # Third rule
            score += 30 * self.countCorners(self.maxPlayer)

        else:
            # First rule
            if self.checkWinner() == -1: return -10000

            # Second rule
            score -= 500 * self.twoInARow(self.minPlayer, '_')
            score -= 100 * self.twoInARow(self.minPlayer, self.maxPlayer)

            if score != 0: return score

            # Third rule
            score -= 30 * self.countCorners(self.minPlayer)

        return score

    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0

        if isMax:
            # First rule
            if self.checkWinner() == 1: return 10000
            
            # Second rule, try those spots where you have more possible winning moves
            score += 100 * self.twoInARow(self.maxPlayer, '_')
            score += 500 * self.twoInARow(self.maxPlayer, self.minPlayer)

            if score != 0: return score

            # Third rule
            score += 30 * self.countCorners(self.maxPlayer)

        else:
            # First rule
            if self.checkWinner() == -1: return -10000

            # Second rule
            score -= 100 * self.twoInARow(self.minPlayer, '_')
            score -= 500 * self.twoInARow(self.minPlayer, self.maxPlayer)

            if score != 0: return score

            # Third rule
            score -= 30 * self.countCorners(self.minPlayer)

        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        for row in self.board:
            for spot in row:
                if spot == '_': return True
        return False

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if minPlayer is the winner.
        """
        #YOUR CODE HERE
        maxWins = self.threeInARow(self.maxPlayer)
        minWins = self.threeInARow(self.minPlayer)
        if maxWins == 1: return 1
        elif minWins == 1: return -1
        else: return 0

    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        self.expandedNodes += 1
        if (depth == self.maxDepth) or (self.checkMovesLeft() == 0) or (self.checkWinner() != 0): 
            return self.evaluatePredifined(not isMax)

        if isMax:
            # For maxPlayer, initialize utility as the smallest possible utility
            bestValue = -self.winnerMaxUtility
            # Try every possible move and maximize the minimal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.maxPlayer
                bestValue = max(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if bestValue >= beta: return bestValue
                alpha = max(bestValue, alpha)

        else: 
            # For minPlayer, initialize utility as the largest possible utility
            bestValue = -self.winnerMinUtility
            # Try every possible move and minimize the maximal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.minPlayer
                bestValue = min(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if bestValue <= alpha: return bestValue
                beta = min(bestValue, beta)

        return bestValue

    def alphabetaOwnAgent(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        self.expandedNodes += 1
        if (depth == self.maxDepth) or (self.checkMovesLeft() == 0) or (self.checkWinner() != 0): 
            return self.evaluateDesigned(not isMax)

        if isMax:
            # For maxPlayer, initialize utility as the smallest possible utility
            bestValue = -self.winnerMaxUtility
            # Try every possible move and maximize the minimal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.maxPlayer
                bestValue = max(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if bestValue >= beta: return bestValue
                alpha = max(bestValue, alpha)

        else: 
            # For minPlayer, initialize utility as the largest possible utility
            bestValue = -self.winnerMinUtility
            # Try every possible move and minimize the maximal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.minPlayer
                bestValue = min(bestValue, self.alphabeta(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, alpha, beta, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'
                if bestValue <= alpha: return bestValue
                beta = min(bestValue, beta)

        return bestValue

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        self.expandedNodes += 1
        if (depth == self.maxDepth) or (self.checkMovesLeft() == 0) or (self.checkWinner() != 0): 
            return self.evaluatePredifined(not isMax)

        if isMax:
            # For maxPlayer, initialize utility as the smallest possible utility
            bestValue = -self.winnerMaxUtility
            # Try every possible move and maximize the minimal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.maxPlayer
                bestValue = max(bestValue, self.minimax(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'

        else: 
            # For minPlayer, initialize utility as the largest possible utility
            bestValue = -self.winnerMinUtility
            # Try every possible move and minimize the maximal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.minPlayer
                bestValue = min(bestValue, self.minimax(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'

        return bestValue

    def minimaxOwnAgent(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        self.expandedNodes += 1
        if (depth == self.maxDepth) or (self.checkMovesLeft() == 0) or (self.checkWinner() != 0): 
            return self.evaluateDesigned(not isMax)

        if isMax:
            # For maxPlayer, initialize utility as the smallest possible utility
            bestValue = -self.winnerMaxUtility
            # Try every possible move and maximize the minimal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.maxPlayer
                bestValue = max(bestValue, self.minimax(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'

        else: 
            # For minPlayer, initialize utility as the largest possible utility
            bestValue = -self.winnerMinUtility
            # Try every possible move and minimize the maximal best value
            possibleMoveOptions = self.possibleMoves(currBoardIdx)
            for moveOption in possibleMoveOptions:
                self.board[moveOption[0]][moveOption[1]] = self.minPlayer
                bestValue = min(bestValue, self.minimax(depth+1, (moveOption[0]%3)*3 + moveOption[1]%3, not isMax))
                self.board[moveOption[0]][moveOption[1]] = '_'

        return bestValue

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        bestValue=[]
        expandedNodes=[]

        # Initialize current player, current game board index, alpha, and beta
        self.currPlayer = maxFirst
        currentBoardIdx = self.startBoardIdx
        alpha = -self.winnerMaxUtility 
        beta = -self.winnerMinUtility
        move = 1

        # While there exists available move and there is no winner
        while ((self.checkMovesLeft() == 1) and (self.checkWinner() == 0)):
            currentBoard = self.globalIdx[currentBoardIdx]
            
            # For maxPlayer
            if self.currPlayer == 1: 
                currentPlayer = self.maxPlayer
                currentBestValue = -self.winnerMaxUtility
            # For minPlayer
            else: 
                currentPlayer = self.minPlayer
                currentBestValue = -self.winnerMinUtility
        
            # Evaluate every possible move in the local game board
            for i in range(3):
                for j in range(3):
                    if self.board[currentBoard[0]+i][currentBoard[1]+j] == '_':
                    
                        self.board[currentBoard[0]+i][currentBoard[1]+j] = currentPlayer
                        
                        # Use minimax if the current player is maxPlayer and it is using minimax
                        # or if the current layer is minPlayer and it is using minimax
                        if (self.currPlayer == 1 and isMinimaxOffensive == 1) or (self.currPlayer == 0 and isMinimaxDefensive == 1):
                            currentValue = self.minimax(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, not self.currPlayer)
                        
                        # Use alpha-beta if the current player is maxPlayer and it is using alpha-beta
                        # or if the current layer is minPlayer and it is using alpha-beta
                        else:
                            currentValue = self.alphabeta(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, alpha, beta, not self.currPlayer)
                    
                        self.board[currentBoard[0]+i][currentBoard[1]+j] = '_'

                        # Update the best value and best move option
                        if (self.currPlayer == 1 and currentValue > currentBestValue) or (self.currPlayer == 0 and currentValue < currentBestValue):
                            currentBestValue = currentValue
                            bestMoveOption = (currentBoard[0]+i, currentBoard[1]+j) 

            # Update the whole game board
            self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer
            gameBoards.append(self.board)
            expandedNodes.append(self.expandedNodes)
            bestMove.append(bestMoveOption)
            bestValue.append(currentBestValue)
            currentBoardIdx = (bestMoveOption[0]%3)*3 + bestMoveOption[1]%3
            self.currPlayer = not self.currPlayer
            self.printGameBoard()
            move += 1

            if (move%10 == 1): print('-----This is the game board after', move, 'st move-----')
            elif (move%10 == 2): print('-----This is the game board after', move, 'nd move-----')
            elif (move%10 == 3): print('-----This is the game board after', move, 'rd move-----')
            else: print('-----This is the game board after', move, 'th move-----')

        winner = self.checkWinner()
        self.printGameBoard()

        return gameBoards, bestMove, bestValue, winner, expandedNodes

    def playGameYourAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        bestValue=[]
        expandedNodes=[]

        # Initialize current player, current game board index, alpha, and beta
        self.currPlayer = maxFirst
        currentBoardIdx = self.startBoardIdx
        alpha = -self.winnerMaxUtility 
        beta = -self.winnerMinUtility
        move = 1

        # While there exists available move and there is no winner
        while ((self.checkMovesLeft() == 1) and (self.checkWinner() == 0)):
            currentBoard = self.globalIdx[currentBoardIdx]
            
            # For maxPlayer
            if self.currPlayer == 1: 
                currentPlayer = self.maxPlayer
                currentBestValue = -self.winnerMaxUtility
            # For minPlayer
            else: 
                currentPlayer = self.minPlayer
                currentBestValue = -self.winnerMinUtility
        
            # Evaluate every possible move in the local game board
            for i in range(3):
                for j in range(3):
                    if self.board[currentBoard[0]+i][currentBoard[1]+j] == '_':
                    
                        self.board[currentBoard[0]+i][currentBoard[1]+j] = currentPlayer
                        
                        # Use minimax if the current player is maxPlayer and it is using minimax
                        if (self.currPlayer == 1 and isMinimaxOffensive == 1):
                            currentValue = self.minimax(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, not self.currPlayer)
                        
                        # Use alpha-beta if the current player is maxPlayer and it is using alpha-beta
                        elif (self.currPlayer == 1 and isMinimaxOffensive == 0):
                            currentValue = self.alphabeta(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, alpha, beta, not self.currPlayer)
                    
                        # Else, if our own agent is using minimax or alpha-beta
                        elif (self.currPlayer == 0 and isMinimaxDefensive == 1):
                            currentValue = self.minimaxOwnAgent(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, not self.currPlayer)
                        else: 
                            currentValue = self.alphabetaOwnAgent(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, alpha, beta, not self.currPlayer)

                        self.board[currentBoard[0]+i][currentBoard[1]+j] = '_'

                        # Update the best value and best move option
                        if (self.currPlayer == 1 and currentValue > currentBestValue) or (self.currPlayer == 0 and currentValue < currentBestValue):
                            currentBestValue = currentValue
                            bestMoveOption = (currentBoard[0]+i, currentBoard[1]+j) 

            # Update the whole game board
            self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer
            gameBoards.append(self.board)
            expandedNodes.append(self.expandedNodes)
            bestMove.append(bestMoveOption)
            bestValue.append(currentBestValue)
            currentBoardIdx = (bestMoveOption[0]%3)*3 + bestMoveOption[1]%3
            self.currPlayer = not self.currPlayer
            self.printGameBoard()
            move += 1

            if (move%10 == 1): print('-----This is the game board after', move, 'st move-----')
            elif (move%10 == 2): print('-----This is the game board after', move, 'nd move-----')
            elif (move%10 == 3): print('-----This is the game board after', move, 'rd move-----')
            else: print('-----This is the game board after', move, 'th move-----')

        winner = self.checkWinner()
        self.printGameBoard()

        return gameBoards, bestMove, bestValue, winner, expandedNodes

    def playGameHuman(self,maxFirst,isHumanOffensive,isAgentMinimax):
        """
        This function implements the processes of the game of your own agent vs a human.
        input args:
        maxFirst: True if maxPlayer plays first
        isHumanOffensive: True if human is offensive
        isAgentMinimax: True if my agent is using minimax
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        expandedNodes=[]

        # Initialize current player, current game board index, alpha, and beta
        self.currPlayer = maxFirst
        currentBoardIdx = self.startBoardIdx
        alpha = -self.winnerMaxUtility 
        beta = -self.winnerMinUtility
        move = 0

        # While there exists available move and there is no winner
        if isHumanOffensive == 1:

            while ((self.checkMovesLeft() == 1) and (self.checkWinner() == 0)):
                currentBoard = self.globalIdx[currentBoardIdx]
            
                # My agent
                if self.currPlayer == 0: 
                    currentPlayer = self.minPlayer
                    currentBestValue = -self.winnerMinUtility
        
                    # Evaluate every possible move in the local game board
                    for i in range(3):
                        for j in range(3):
                            if self.board[currentBoard[0]+i][currentBoard[1]+j] == '_':
                    
                                self.board[currentBoard[0]+i][currentBoard[1]+j] = currentPlayer
                        
                                # Use minimax if my own agent is using minimax
                                if isAgentMinimax == 1:
                                    currentValue = self.minimaxOwnAgent(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, not self.currPlayer)
                        
                                # Use alpha-beta if my own agent is using alpha-beta
                                else: currentValue = self.alphabetaOwnAgentself.alphabeta(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, alpha, beta, not self.currPlayer)
                    
                                self.board[currentBoard[0]+i][currentBoard[1]+j] = '_'

                                # Update the best value and best move option
                                if (self.currPlayer == 1 and currentValue > currentBestValue) or (self.currPlayer == 0 and currentValue < currentBestValue):
                                    currentBestValue = currentValue
                                    bestMoveOption = (currentBoard[0]+i, currentBoard[1]+j) 

                    # Update the whole game board
                    self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer
            
                else: # Human
                    currentPlayer = self.maxPlayer
                    self.printGameBoard()
                    print("Current local board has index", currentBoardIdx)
                    X = int(input("Enter X (0, 1, 2) coordinate:")) 
                    Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    while((X not in self.validSet) or (Y not in self.validSet) or (self.board[currentBoard[0]+X][currentBoard[1]+Y] != '_')):
                        X = int(input("Wrong, enter X (0, 1, 2) coordinate:")) 
                        Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    bestMoveOption = (currentBoard[0]+X, currentBoard[1]+Y)

                    # Update the whole game board
                    self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer

                gameBoards.append(self.board)
                expandedNodes.append(self.expandedNodes)
                bestMove.append(bestMoveOption)
                currentBoardIdx = (bestMoveOption[0]%3)*3 + bestMoveOption[1]%3
                self.currPlayer = not self.currPlayer
                self.printGameBoard()
                move += 1

                if (move%10 == 1): print('-----This is the game board after', move, 'st move-----')
                elif (move%10 == 2): print('-----This is the game board after', move, 'nd move-----')
                elif (move%10 == 3): print('-----This is the game board after', move, 'rd move-----')
                else: print('-----This is the game board after', move, 'th move-----')
        
            winner = self.checkWinner()
            self.printGameBoard()

            return gameBoards, bestMove, winner, expandedNodes
            
        else:

            while ((self.checkMovesLeft() == 1) and (self.checkWinner() == 0)):
                currentBoard = self.globalIdx[currentBoardIdx]
            
                # My agent
                if self.currPlayer == 1: 
                    currentPlayer = self.maxPlayer
                    currentBestValue = -self.winnerMaxUtility
        
                    # Evaluate every possible move in the local game board
                    for i in range(3):
                        for j in range(3):
                            if self.board[currentBoard[0]+i][currentBoard[1]+j] == '_':
                    
                                self.board[currentBoard[0]+i][currentBoard[1]+j] = currentPlayer
                        
                                # Use minimax if my own agent is using minimax
                                if isAgentMinimax == 1:
                                    currentValue = self.minimaxOwnAgent(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, not self.currPlayer)
                        
                                # Use alpha-beta if my own agent is using alpha-beta
                                else: currentValue = self.alphabetaOwnAgentself.alphabeta(0, ((currentBoard[0]+i)%3)*3 + (currentBoard[1]+j)%3, alpha, beta, not self.currPlayer)
                    
                                self.board[currentBoard[0]+i][currentBoard[1]+j] = '_'

                                # Update the best value and best move option
                                if (self.currPlayer == 1 and currentValue > currentBestValue) or (self.currPlayer == 0 and currentValue < currentBestValue):
                                    currentBestValue = currentValue
                                    bestMoveOption = (currentBoard[0]+i, currentBoard[1]+j) 

                    # Update the whole game board
                    self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer
            
                else: # Human
                    currentPlayer = self.minPlayer
                    self.printGameBoard()
                    print("Current local board has index", currentBoardIdx)
                    X = int(input("Enter X (0, 1, 2) coordinate:")) 
                    Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    while(((X not in self.validSet) or (Y not in self.validSet)) or (self.board[currentBoard[0]+X][currentBoard[1]+Y] != '_')):
                        X = int(input("Wrong, enter X (0, 1, 2) coordinate:")) 
                        Y = int(input("Enter Y (0, 1, 2) coordinate:"))
                    bestMoveOption = (currentBoard[0]+X, currentBoard[1]+Y)

                    # Update the whole game board
                    self.board[bestMoveOption[0]][bestMoveOption[1]] = currentPlayer
            
                gameBoards.append(self.board)
                expandedNodes.append(self.expandedNodes)
                bestMove.append(bestMoveOption)
                currentBoardIdx = (bestMoveOption[0]%3)*3 + bestMoveOption[1]%3
                self.currPlayer = not self.currPlayer
                self.printGameBoard()
                move += 1

                if (move%10 == 1): print('-----This is the game board after', move, 'st move-----')
                elif (move%10 == 2): print('-----This is the game board after', move, 'nd move-----')
                elif (move%10 == 3): print('-----This is the game board after', move, 'rd move-----')
                else: print('-----This is the game board after', move, 'th move-----')

            winner = self.checkWinner()
            self.printGameBoard()

            return gameBoards, bestMove, winner, expandedNodes

if __name__=="__main__":
    uttt=ultimateTicTacToe()

    # Part I:
    # maxPlayer goes first: offensive(minimax) vs defensive(minimax), offensive(minimax) vs defensive(alpha-beta) 
    # gameBoards, bestMove, bestValue, winner, expandedNodes=uttt.playGamePredifinedAgent(True,True,True)
    # gameBoards, bestMove, bestValue, winner, expandedNodes=uttt.playGamePredifinedAgent(True,True,False)

    # minPlayer goes first: offensive(alpha-beta) vs defensive(minimax), and offensive(alpha-beta) vs defensive(alpha-beta)
    gameBoards, bestMove, bestValue, winner, expandedNodes=uttt.playGamePredifinedAgent(False,False,True)
    # gameBoards, bestMove, bestValue, winner, expandedNodes=uttt.playGamePredifinedAgent(False,False,False)

    # Part II:
    # gameBoards, bestMove, bestValue, winner, expandedNodes=uttt.playGameYourAgent(True,True,True)

    # Part III:
    # maxPlayer goes first, human is offensive, my agent uses minimax
    # gameBoards, bestMove, winner, expandedNodes=uttt.playGameHuman(True,True,True)
    # maxPlayer goes first, human is defensive, my agent uses minimax
    # gameBoards, bestMove, winner, expandedNodes=uttt.playGameHuman(True,False,True)
   

    print(expandedNodes)
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
