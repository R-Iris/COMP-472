# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python
import time
import random
import numpy as np
import collections

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    E1 = 4
    E2 = 5

    winner = 'None'

    currentStatesD1 = 0
    currentStatesD2 = 0
    currentT = 0.0
    miniMaxStartT = 0.0
    alphaBetaStartT = 0.0

    heuE1Counter = 0
    heuE2Counter = 0
    heuTOTAL = 0
    AVGStates = 0.0

    dictDepthD1 = {}
    dictDepthD2 = {}
    dictTOTALbyDepth = {}

    numberOfMoves = 0

    listOfTimes = []

    def __init__(self, n=3, b=0, bboard=[], s=3, d1=0, d2=0, t=0.0):
        self.n = n
        self.b = b
        self.bboard = bboard
        self.s = s
        self.d1 = d1  # M ax value of d1
        self.d2 = d2  # Max value of d2
        self.t = t  # Max value of t
        self.initialize_game()
        self.file = open(F'gameTrace-{self.n}{self.b}{self.s}{self.t}.txt', 'w')

        # Writing the first line to the file
        self.file.write(F'n={self.n} b={self.b} s={self.s} t={self.t}\n')
        self.file.write(F'blocs={self.bboard}\n\n')

    def initialize_game(self):
        self.current_state = [['.' for x in range(self.n)] for y in range(self.n)]

        if self.bboard == []:
            for i in range(self.b):
                X = random.randint(0, self.n - 1)
                Y = random.randint(0, self.n - 1)
                while self.current_state[X][Y] == '#':
                    X = random.randint(0, self.n - 1)
                    Y = random.randint(0, self.n - 1)
                self.current_state[X][Y] = '#'
                self.bboard.append([X, Y])
        else:
            for element in self.bboard:
                self.current_state[element[0]][element[1]] = '#'

        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self):
        print()
        self.file.write('\n\t')
        for z in range(0, self.n):
            self.file.write(F' {chr(z + 65)}')
        self.file.write('\n')
        for y in range(0, self.n):
            self.file.write(F' {y} | ')
            for x in range(0, self.n):
                print(F'{self.current_state[x][y]}', end="")
                self.file.write(F'{self.current_state[x][y]} ')
            print()
            self.file.write('\n')
        print()
        self.file.write('\n\n')

    def is_valid(self, px, py):
        if px < 0 or px > self.n - 1 or py < 0 or py > self.n - 1:
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def is_end(self):
        # Vertical win
        for x in range(0, self.n):  # Once for each column
            for y in range(0,
                           self.n - self.s + 1):  # range is -s for efficiency reasons and out of bounds("beginning of the snake")
                breakX = False
                if self.current_state[x][y] == '#' or self.current_state[x][y] == '.':
                    continue
                for s in range(0, self.s - 1):  # -1 to the range, for s = 3, there are 2 checks -> 1 = 2, 2 = 3
                    if self.current_state[x][y + s] != self.current_state[x][y + s + 1]:
                        breakX = True
                        break  # break the s loop
                if breakX:
                    continue
                return self.current_state[x][y]  # win with (x, y)

        # Horizontal win
        for y in range(0, self.n):  # Once for each row
            for x in range(0,
                           self.n - self.s + 1):  # range is -s for efficiency reasons and out of bounds("beginning of the snake")
                breakY = False
                if self.current_state[x][y] == '#' or self.current_state[x][y] == '.':
                    continue
                for s in range(0, self.s - 1):  # -1 to the range, for s = 3, there are 2 checks -> 1 = 2, 2 = 3
                    if self.current_state[x + s][y] != self.current_state[x + s + 1][y]:
                        breakY = True
                        break  # break the s loop
                if breakY:
                    continue
                return self.current_state[x][y]  # win with (x, y)

        # Top left to bottom right diagonals
        for x in range(0, self.n - self.s + 1):  # Till n-s+1 columns (exclusive)
            for y in range(0,
                           self.n - self.s + 1):  # range is -s for efficiency reasons and out of bounds("beginning of the snake")
                break_LR_Diag = False
                if self.current_state[x][y] == '#' or self.current_state[x][y] == '.':
                    continue
                for s in range(0, self.s - 1):  # -1 to the range, for s = 3, there are 2 checks -> 1 = 2, 2 = 3
                    if self.current_state[x + s][y + s] != self.current_state[x + s + 1][
                        y + s + 1]:  # to find the next element we add (1,1) to our current
                        break_LR_Diag = True
                        break  # break from s loop
                if break_LR_Diag:
                    continue
                return self.current_state[x][y]  # win with (x, y)

        # Top right to bottom left diagonals
        for x in range(self.s - 1, self.n):  # Start at n-s'th column till n (Excluded)
            for y in range(0,
                           self.n - self.s + 1):  # range is - s for efficiency reasons and out of bounds ("beginning of the snake" )
                break_RL_Diag = False
                if self.current_state[x][y] == '#' or self.current_state[x][y] == '.':
                    continue
                for s in range(0, self.s - 1):  # -1 to the range, for s = 3, there are 2 checks -> 1 = 2, and 2 = 3
                    if self.current_state[x - s][y + s] != self.current_state[x - s - 1][
                        y + s + 1]:  # x decrements but y still increments
                        break_RL_Diag = True
                        break  # break from s loop
                if break_RL_Diag:
                    continue
                return self.current_state[x][y]  # win with (x, y)

        # Is whole board full?
        for i in range(0, self.n):
            for j in range(0, self.n):
                # There's an empty field, we continue the game
                if self.current_state[i][j] == '.':
                    return None
        # It's a tie!

        return '.'

    # adapted for new ruleset
    def check_end(self):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result != None:
            if self.result == 'X':
                self.winner = 'X'
                print('The winner is X!')
                self.file.write('\nThe winner is X!\n')

            elif self.result == 'O':
                self.winner = 'O'
                print('The winner is O!')
                self.file.write('\nThe winner is O!\n')

            elif self.result == '.':
                self.winner = 'tie'
                print("It's a tie!")
                self.file.write("\nIt's a tie!\n")

            self.file.write(F'6(b)i\tAverage evaluation time: {round(np.array(self.listOfTimes).mean(),3)}s\n')
            self.file.write(F'(b)ii\tTotal heuristic evaluations: {self.heuTOTAL}\n')
            self.file.write(F'6(b)iii\tEvaluations by depth: {self.dictTOTALbyDepth}\n')
            self.file.write(F'6(b)iv\tAverage evaluation depth: {round(self.AVGStates/self.numberOfMoves, 3)}\n')
            self.file.write(F'6(b)vi\tTotal number of moves: {self.numberOfMoves}\n')

            self.initialize_game()
        return self.result

    # looks fine
    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = input('enter the x coordinate: ')
            px = ord(px) - 65
            py = int(input('enter the y coordinate: '))
            if self.is_valid(px, py):
                return (px, py)
            else:
                print('The move is not valid! Try again.')

    # looks fine
    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn

    # need to adapt to new ruleset
    def minimax(self, max=False, depth=0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:

        # Updating the timer at intervals every time the minimax() method is called
        miniMaxEndT = time.time()
        self.currentT = round(miniMaxEndT - self.miniMaxStartT, 7)

        value = 2
        if max:
            value = -2
        x = None
        y = None

        # Evaluation Function
        result = self.is_end()
        if result == 'X':
            return (-1, x, y)
        elif result == 'O':
            return (1, x, y)
        elif result == '.':
            return (0, x, y)

        # only changes involve the limits of the range of i and j,
        # the rest is minimax choices between higher/lower values
        for i in range(0, self.n):  # change hardcoded 3 to self.n
            for j in range(0, self.n):  # change hardcoded 3 to self.n

                if depth == 0:
                    # or self.is_end():
                    return (value, x, y)

                #if (self.currentStatesD1 >= self.d1 != 0) or (self.currentStatesD2 >= self.d2 != 0):

                if self.currentT >= self.t != 0:
                    print(F'\nPlayer {self.player_turn} under AI control has taken too long to decide.'
                          F'\nPlayer {self.switch_player()} has won!')
                    self.file.write(F'\nPlayer {self.player_turn} under AI control has taken too long to decide.'
                                    F'\nPlayer {self.switch_player()} has won!')
                    exit(0)

                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        if depth == self.d2:
                            x = i
                            y = j
                        # d2 is the 2nd player because O starts second
                        self.currentStatesD2 += 1

                        # print("D2 depth: " + str(self.currentD2))
                        (v, _, _) = self.minimax(max=False, depth=depth-1)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        if depth == self.d1:
                            x = i
                            y = j
                        # d1 is the 1st player because X starts first
                        self.currentStatesD1 += 1

                        # print("D1 depth: " + str(self.currentD1))
                        (v, _, _) = self.minimax(max=True, depth=depth-1)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
        return (value, x, y)

    def alphabeta(self, alpha=-10000, beta=10000, max=False, heur=None, depth=0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:

        # Updating the timer at intervals every time the alphabeta() method is called
        alphaBetaEndT = time.time()
        self.currentT = round(alphaBetaEndT - self.alphaBetaStartT, 7)

        # Current dictionary with depths as keys and states visited as values
        if max and self.currentStatesD2 != 0:
            currentDepthDect = {depth: self.currentStatesD2}
            # If we want to maximize, then we're looking at D2 because O maximizes
            self.dictDepthD2.update(currentDepthDect)
        elif self.currentStatesD1 != 0:
            currentDepthDect = {depth: self.currentStatesD1}
            # If we want to minimize, then we're looking at D1 because X minimizes
            self.dictDepthD1.update(currentDepthDect)

        value = 10000
        if max:
            value = -10000
        x = None
        y = None

        # Evaluation Function
        result = self.is_end()
        if result == 'X':
            return (-1, x, y)
        elif result == 'O':
            return (1, x, y)
        elif result == '.':
            return (0, x, y)

        for i in range(0, self.n):
            # only change to these for loops is the limits of each range for i and j
            # functionality for alpha-beta remains the same
            for j in range(0, self.n):

                if depth == 0:
                    return (value, x, y)

                if self.currentT >= self.t != 0:
                    print(F'\nPlayer {self.player_turn} under AI control has taken too long to decide.'
                          F'\nPlayer {self.switch_player()} has won!')
                    exit(0)

                if self.current_state[i][j] == '.':
                    if heur == self.E1:
                        value = self.e1(x=i, y=j)
                    elif heur == self.E2:
                        value = self.e2(x=i, y=j)
                    if max:
                        self.current_state[i][j] = 'O'
                        self.currentStatesD2 += 1
                        (v, k, l) = self.alphabeta(alpha, beta, max=False, heur=heur, depth=depth-1)
                        # this will set the value to v and the coordinate to the best possible child
                        # will then go back up the recursion ladder and compare with another possibility
                        if v > value and k is not None and l is not None:
                            # print(" i: " + str(i) + " j: " + str(j) +  " value: " + str(value) + " k: " + str(k) + " l: " + str(l) + " v: " + str(v))
                            value = v
                            x = k
                            y = l
                        else:  # choose the current value and coordinate (since above is invalid)
                            # so it will go back up the recursion ladder and compare with another possibility
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        self.currentStatesD1 += 1
                        (v, k, l) = self.alphabeta(alpha, beta, max=True, heur=heur, depth=depth-1)
                        # this will set the value to v and the coordinate to the best possible child
                        # will then go back up the recursion ladder and compare with another possibility
                        if v < value and k is not None and l is not None:
                            # print(" i: " + str(i) + " j: " + str(j) +  " value: " + str(value) + " k: " + str(k) + " l: " + str(l) + " v: " + str(v))
                            value = v
                            x = k
                            y = l
                        else:  # choose the current value and coordinate (since above is invalid)
                            # so it will go back up the recursion ladder and compare with another possibility
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
                    if max:
                        if value >= beta:
                            return (value, x, y)
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return (value, x, y)
                        if value < beta:
                            beta = value
        return (value, x, y)

    def e1(self, x, y):
        self.heuE1Counter += 1
        self.heuTOTAL +=1

        score = 0
        symbol = self.current_state[x][y]
        # 4 cases: Horizontal, Vertical, TL, TR
        # For each case:- 2 directions

        # Horizontal
        # -----------
        # Right
        i = 0
        snakeCount = 0
        while y + i < self.n:
            ithSymbol = self.current_state[x][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # Left
        i = 0
        while y - i >= 0:
            ithSymbol = self.current_state[x][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # If snakeCount adds up to s (can make a snake on this line)
        if snakeCount >= self.s:
            score += snakeCount * 4
        else:
            score += snakeCount * 2
        # -----------
        # Vertical
        # Down
        i = 0
        snakeCount = 0
        while x + i < self.n:
            ithSymbol = self.current_state[x + i][y]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # Up
        i = 0
        while x - i >= 0:
            ithSymbol = self.current_state[x - i][y]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        if snakeCount >= self.s:
            score += snakeCount * 4
        else:
            score += snakeCount * 2
        # -----------
        # Favours Diagonals
        # TL
        # BR
        i = 0
        snakeCount = 0
        while x + i < self.n and y + i < self.n:
            ithSymbol = self.current_state[x + i][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # TL
        i = 0
        while x - i >= 0 and y - i >= 0:
            ithSymbol = self.current_state[x - i][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        if snakeCount >= self.s:
            score += snakeCount * 20
        else:
            score += snakeCount * 8
        # -----------
        # TR

        # BL
        i = 0
        snakeCount = 0
        while x - i >= 0 and y + i < self.n:
            ithSymbol = self.current_state[x - i][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # TR
        i = 0
        while x + i < self.n and y - i >= 0:
            ithSymbol = self.current_state[x + i][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        if snakeCount >= self.s:
            score += snakeCount * 20
        else:
            score += snakeCount * 8
        # -----------
        return score

    def e2(self, x, y):
        self.heuE2Counter += 1
        self.heuTOTAL += 1

        score = 0
        symbol = self.current_state[x][y]
        # 4 cases: Horizontal, Vertical, TL, TR
        # For each case:- 2 directions

        # Horizontal
        # -----------
        # Precondition
        # Right
        i = 0
        snakeCount = 0
        while y + i < self.n:
            ithSymbol = self.current_state[x][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # Left
        i = 0
        while y - i >= 0:
            ithSymbol = self.current_state[x][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # If snakeCount adds up to s (can make a snake on this line)
        if snakeCount >= self.s:
            # Right Score Calculation
            i = 0
            while y + i < self.n:
                ithSymbol = self.current_state[x][y + i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
            # Left Score Calculation
            i = 0
            while y - i >= 0:
                ithSymbol = self.current_state[x][y - i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
        # -----------
        # Vertical
        # Precondition
        # Down
        i = 0
        snakeCount = 0
        while x + i < self.n:
            ithSymbol = self.current_state[x + i][y]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # Up
        i = 0
        while x - i >= 0:
            ithSymbol = self.current_state[x - i][y]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # If snakeCount adds up to s (can make a snake on this line)
        if snakeCount >= self.s:
            # Down Score Calculation
            i = 0
            while x + i < self.n:
                ithSymbol = self.current_state[x + i][y]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
            # Up Score Calculation
            i = 0
            while x - i >= 0:
                ithSymbol = self.current_state[x - i][y]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
        # -----------
        # TL
        # Precondition
        # BR
        i = 0
        snakeCount = 0
        while x + i < self.n and y + i < self.n:
            ithSymbol = self.current_state[x + i][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # TL
        i = 0
        while x - i >= 0 and y - i >= 0:
            ithSymbol = self.current_state[x - i][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # If snakeCount adds up to s (can make a snake on this line)
        if snakeCount >= self.s:
            # BR Score Calculation
            i = 0
            while x + i < self.n and y + i < self.n:
                ithSymbol = self.current_state[x + i][y + i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
            # TL Score Calculation
            i = 0
            while x - i >= 0 and y - i >= 0:
                ithSymbol = self.current_state[x - i][y - i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
        # -----------
        # TR
        # Precondition
        # BL
        i = 0
        snakeCount = 0
        while x - i >= 0 and y + i < self.n:
            ithSymbol = self.current_state[x - i][y + i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # TR
        i = 0
        while x + i < self.n and y - i >= 0:
            ithSymbol = self.current_state[x + i][y - i]
            if ithSymbol == symbol or ithSymbol == '.':
                snakeCount += 1
                i += 1
            else:
                break
        # If snakeCount adds up to s (can make a snake on this line)
        if snakeCount >= self.s:
            # BL Score Calculation
            i = 0
            while x - i >= 0 and y + i < self.n:
                ithSymbol = self.current_state[x - i][y + i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
            # TR Score Calculation
            i = 0
            while x + i < self.n and y - i >= 0:
                ithSymbol = self.current_state[x + i][y - i]
                if ithSymbol == '#':
                    break
                elif ithSymbol == '.':
                    if self.player_turn == 'O':
                        score += 1
                    else:
                        score -= 1
                elif ithSymbol == symbol:
                    if self.player_turn == 'O':
                        score += self.n - i
                    else:
                        score -= self.n - i
                else:
                    break
                i += 1
        # -----------
        return score

    def play(self, algo=None, player_x=None, player_o=None, heur_x=None, heur_o=None):
        if player_x == self.HUMAN:
            self.file.write(F'Player 1: HUMAN\n')
        else:
            self.file.write(F'Player 1: AI d={self.d1} a={"True" if algo is not None else "False"} '
                            F'{"Type=ALPHABETA" if algo == self.ALPHABETA else "Type=MINIMAX"}'
                            F'{" e1(regular)" if heur_x == self.E1 else "e2(defensive)"}\n')
        if player_o == self.HUMAN:
            self.file.write(F'Player 2: HUMAN')
        else:
            self.file.write(F'Player 2: AI d={self.d2} a={"True" if algo is not None else "False"} '
                            F'{"Type=ALPHABETA" if algo == self.ALPHABETA else "Type=MINIMAX"}'
                            F'{" e1(regular)" if heur_x == self.E1 else "e2(defensive)"}\n')

        if algo == None:
            algo = self.ALPHABETA
        if heur_x == None:
            heur_x = self.E1
        if heur_o == None:
            heur_o = self.E2
        if player_x == None:
            player_x = self.HUMAN
        if player_o == None:
            player_o = self.HUMAN

        # A while loop is a move
        while True:
            self.draw_board()
            if self.check_end():
                return

            # Resetting the d1, d2 and time
            self.currentStatesD1 = 0
            self.currentStatesD2 = 0
            self.currentT = 0.0
            self.miniMaxStartT = 0.0
            self.alphaBetaStartT = 0.0
            self.heuE1Counter = 0
            self.heuE2Counter = 0

            if (self.player_turn == 'X' and player_x == self.HUMAN) or (
                    self.player_turn == 'O' and player_o == self.HUMAN):
                (x, y) = self.input_move()
                self.file.write(F'Player {self.player_turn} under HUMAN control plays: {chr(x + 65)} {y}\n')
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                start = time.time()
                if algo == self.MINIMAX:
                    self.miniMaxStartT = time.time()
                    if self.player_turn == 'X':
                        (_, x, y) = self.minimax(max=False, depth=self.d1)
                    else:
                        (_, x, y) = self.minimax(max=True, depth=self.d2)
                else:  # algo == self.ALPHABETA:
                    self.alphaBetaStartT = time.time()
                    if self.player_turn == 'X':
                        (m, x, y) = self.alphabeta(max=False, heur=heur_x, depth=self.d1)
                    else:
                        (m, x, y) = self.alphabeta(max=True, heur=heur_o, depth=self.d2)
                end = time.time()

                #print("Current depth: " + str(self.depth))
                print("States checked for P1: " + str(self.currentStatesD1) + " || States checked for P2: " + str(self.currentStatesD2))
                self.listOfTimes.append(round(end - start, 7))
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
                self.file.write(F'\nPlayer {self.player_turn} under AI control plays:  {chr(x + 65)} {y}\n\n')
                self.file.write(
                    F'i Evaluation time: {round(end - start, 7)}s\n')
                if self.player_turn == 'X':
                    self.file.write(F'ii Heuristic evaluations: {self.heuE1Counter if heur_x==self.E1 else self.heuE2Counter}\n')
                    self.file.write(F'iii Evaluations by depth: {dict(sorted(self.dictDepthD1.items()))}\n')
                    self.file.write(F'iv Average evaluation depth: {round(np.array(list(self.dictDepthD1.keys())).mean(), 3)}\n')
                    self.AVGStates += round(np.array(list(self.dictDepthD1.keys())).mean(), 3)
                    for key, value in self.dictDepthD1.items():
                        if key not in self.dictTOTALbyDepth:
                            self.dictTOTALbyDepth[key] = value
                        else:
                            self.dictTOTALbyDepth[key] += value

                else:
                    self.file.write(F'ii Heuristic evaluations: {self.heuE1Counter if heur_o==self.E1 else self.heuE2Counter}\n')
                    self.file.write(F'iii Evaluations by depth: {dict(sorted(self.dictDepthD2.items()))}\n')
                    self.file.write(F'iv Average evaluation depth: {round(np.array(list(self.dictDepthD2.keys())).mean(), 3)}\n')
                    self.AVGStates += round(np.array(list(self.dictDepthD1.keys())).mean(), 3)
                    for key, value in self.dictDepthD2.items():
                        if key not in self.dictTOTALbyDepth:
                            self.dictTOTALbyDepth[key] = value
                        else:
                            self.dictTOTALbyDepth[key] += value

            self.current_state[x][y] = self.player_turn
            self.numberOfMoves += 1
            self.switch_player()


def main():
    def scoreboardAppender(algo=None, n=3, b=0, bboard=[], s=3, d1=0, d2=0, t=0.0, r=1):

        winsE1 = 0
        winsE2 = 0
        avgEvalTime = 0
        totalHeurEval = 0
        #evalByDepth = 0
        avgEvalDepth = 0
        avgRecDepth = 0
        avgMovesPerGame = 0

        # print(self.dictTOTALbyDepth)
        # print(self.dictDepthD1)
        # print(self.dictDepthD2)
        # self.file.write(F'6(b)iii\tEvaluations by depth: {self.dictTOTALbyDepth}\n')

        # Create game using args
        g = Game(n=n, b=b, bboard=bboard, s=s, d1=d1, d2=d2, t=t)
        # Play 1xr games with X E1 O E2
        for i in range(r):
            g.play(algo=algo, player_x=Game.AI, player_o=Game.AI, heur_x=Game.E1, heur_o=Game.E2)
            if g.winner == 'X':
                winsE1 += 1
            elif g.winner == 'O':
                winsE2 += 1
            avgEvalTime += round(np.array(g.listOfTimes).mean(),3)
            totalHeurEval += g.heuTOTAL
            avgEvalDepth += round(g.AVGStates/g.numberOfMoves, 3)
            avgRecDepth += 0
            avgMovesPerGame += g.numberOfMoves
        # Play 1xr games with O E1 X E2
        for i in range(r):
            g.play(algo=algo, player_x=Game.AI, player_o=Game.AI, heur_x=Game.E2, heur_o=Game.E1)
            if g.winner == 'X':
                winsE2 += 1
            elif g.winner == 'O':
                winsE1 += 1
            avgEvalTime += round(np.array(g.listOfTimes).mean(),3)
            totalHeurEval += g.heuTOTAL
            avgEvalDepth += round(g.AVGStates/g.numberOfMoves, 3)
            avgRecDepth += 0
            avgMovesPerGame += g.numberOfMoves

        avgEvalTime = avgEvalTime / (2*r)
        avgEvalDepth = avgEvalDepth / (2*r)
        avgRecDepth = avgRecDepth / (2*r)
        avgMovesPerGame = avgMovesPerGame / (2*r)

        # Game Parameters
        file = open('scoreboard.txt', 'a')
        file.write(F'n={g.n} b={g.b} s={g.s} t={g.t}\n')

        # Player Parameters
        file.write(F'Player 1: d={g.d1} a={algo}\n')
        file.write(F'Player 2: d={g.d2} a={algo}\n')

        # Number of Games Played
        file.write(F'{2*r} games\n')

        # The number and percentage of wins for heuristic e1 and for heuristic e2
        file.write(F'Total wins for heuristic e1: {winsE1} ({winsE1/(2*r) *100}%) (Simple)\n')
        file.write(F'Total wins for heuristic e2: {winsE2} ({winsE2/(2*r) *100}%) (Complex)\n')
        #Total wins for heuristic e1: 3 (75.0%) (regular)
        #Total wins for heuristic e2: 1 (25.0%) (defensive)

        # All the information displayed at the end of a game (see Section 2.5.1), but averaged over 2×s games
        #i   Average evaluation time: 4.66s
        file.write(F'Average evaluation time: {round(avgEvalTime,3)}s\n')
        #ii  Total heuristic evaluations: 2100904
        file.write(F'Total heuristic evaluations: {totalHeurEval}\n')
        #iii Evaluations by depth: {6: 1212870, 5: 884185, 4: 3373, 3: 410, 2: 60, 1: 6}
        file.write(F'Evaluations by depth: ????\n')
        #iv  Average evaluation depth: 5.6
        file.write(F'Average evaluation depth: {avgEvalDepth}\n')
        #v   Average recursion depth: 2.9
        file.write(F'Average recursion depth: {avgRecDepth}\n')
        #vi  Average moves per game: 9.75
        file.write(F'Average moves per game: {avgMovesPerGame}\n\n')

    scoreboardAppender(algo=Game.MINIMAX, n=4, b=4, bboard=[[0,0], [0,3], [3,0], [3,3]], s=3, d1=6, d2=6, t=5, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=4, b=4, bboard=[[0,0], [0,3], [3,0], [3,3]], s=3, d1=6, d2=6, t=1, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=5, b=4, bboard=[], s=4, d1=2, d2=6, t=1, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=5, b=4, bboard=[], s=4, d1=6, d2=6, t=5, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=8, b=5, bboard=[], s=5, d1=2, d2=6, t=1, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=8, b=5, bboard=[], s=5, d1=2, d2=6, t=5, r=5)
    #scoreboardAppender(algo=Game.ALPHABETA, n=8, b=6, bboard=[], s=5, d1=6, d2=6, t=1, r=5)
    scoreboardAppender(algo=Game.ALPHABETA, n=8, b=6, bboard=[], s=5, d1=6, d2=6, t=5, r=5)

    #g = Game(n=5, b=1, s=4, d1=4, d2=8, t=9)
    #g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI, heur_x=Game.E2, heur_o=Game.E2)

if __name__ == "__main__":
    main()
