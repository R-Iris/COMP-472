# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time
import random


# n: initgame, isend, minimax, alphabeta

# blocs:
# b: blocs are #'s, b number is used if bboard isnt defined; used in initgame, checkend(X)
# bboard : randomized coordinates if not defined, based on b number; used in initgame, checkend(X)
# bboard : if number of b and coordinates are specified in params, then ignore the number of b

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    E1 = 4
    E2 = 5

    currentD1 = 0
    currentD2 = 0
    currentT = 0.0
    miniMaxStartT = 0.0
    alphaBetaStartT = 0.0

    def __init__(self, n=3, b=0, bboard=[], s=3, d1=0, d2=0, t=0.0):
        self.n = n
        self.b = b
        self.bboard = bboard
        self.s = s
        self.d1 = d1  # M ax value of d1
        # self.currentD1 = 0
        self.d2 = d2  # Max value of d2
        # self.currentD2 = 0
        self.t = t  # Max value of t
        # self.currentT = 0
        self.initialize_game()

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
        else:
            for element in self.bboard:
                self.current_state[element[0]][element[1]] = '#'

        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self):
        print()
        for y in range(0, self.n):
            for x in range(0, self.n):
                print(F'{self.current_state[x][y]}', end="")
            print()
        print()

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
                print('The winner is X!')
            elif self.result == 'O':
                print('The winner is O!')
            elif self.result == '.':
                print("It's a tie!")
            self.initialize_game()
        return self.result

    # looks fine
    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = int(input('enter the x coordinate: '))
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
    def minimax(self, max=False):
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

                if (self.currentD1 >= self.d1 != 0) or (self.currentD2 >= self.d2 != 0):
                    # or self.is_end():
                    return (value, x, y)

                if self.currentT >= self.t != 0:
                    print(F'\nPlayer {self.player_turn} under AI control has taken too long to decide.'
                          F'\nPlayer {self.switch_player()} has won!')
                    exit(0)

                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        # d2 is the 2nd player because O starts second
                        self.currentD2 += 1

                        # print("D2 depth: " + str(self.currentD2))
                        (v, _, _) = self.minimax(max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        # d1 is the 1st player because X starts first
                        self.currentD1 += 1

                        # print("D1 depth: " + str(self.currentD1))
                        (v, _, _) = self.minimax(max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'

        return (value, x, y)

    def alphabeta(self, alpha=-2, beta=2, max=False, heur=None):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:

        # Updating the timer at intervals every time the alphabeta() method is called
        alphaBetaEndT = time.time()
        self.currentT = round(alphaBetaEndT - self.alphaBetaStartT, 7)

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
        
        for i in range(0, self.n):
            # only change to these for loops is the limits of each range for i and j
            # functionality for alpha-beta remains the same
            for j in range(0, self.n):

                if (self.currentD1 >= self.d1 != 0) or (self.currentD2 >= self.d2 != 0):
                    # or self.is_end():
                    return (value, x, y)

                if self.currentT >= self.t != 0:
                    print(F'\nPlayer {self.player_turn} under AI control has taken too long to decide.'
                          F'\nPlayer {self.switch_player()} has won!')
                    exit(0)

                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        self.currentD2 += 1
                        (v, k, l) = self.alphabeta(alpha, beta, max=False)
                        if v > value:
                            if heur == self.E1:
                                if self.e1(i, j) > self.e1(k, l):
                                    value = v
                                    x = i
                                    y = j
                                else:
                                    x = k
                                    y = l
                            elif heur == self.E2:
                                print("hello :)")
                                #e2(x,y)
                            else:
                                value = v
                                x = i
                                y = j

                    else:
                        self.current_state[i][j] = 'X'
                        self.currentD1 += 1
                        (v, k, l) = self.alphabeta(alpha, beta, max=True)
                        if v < value:
                            if heur == self.E1:
                                if self.e1(i, j) < self.e1(k, l):
                                    value = v
                                    x = i
                                    y = j
                                else:
                                    x = k
                                    y = l
                            elif heur == self.E2:
                                print("hello :)")
                                #e2(x,y)
                            else:
                                value = v
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

# WHERE WE LEFT OFF, FOR SOME REASON IT READS X/Y AS NONE
    def e1(self, x, y):
        score = 0
        val = self.current_state[x][y]
        if x+1 < self.n and self.current_state[x+1][y] == val:
            score += 1
        if x-1 > 0 and self.current_state[x-1][y] == val:
            score += 1
        if y+1 < self.n and self.current_state[x][y+1] == val:
            score += 1
        if y-1 > 0 and self.current_state[x][y-1] == val:
            score += 1
        if y+1 < self.n and x+1 < self.n and self.current_state[x+1][y+1] == val:
            score += 2
        if y-1 > 0 and x-1 > 0 and self.current_state[x-1][y-1] == val:
            score += 2
        if x-1 > 0 and y+1 < self.n and self.current_state[x-1][y+1] == val:
            score += 2
        if x+1 < self.n and y-1 > 0 and self.current_state[x+1][y-1] == val:
            score += 2
        return score

    #def e2(self, x, y):


    def play(self, algo=None, player_x=None, player_o=None, heur=None):
        if algo == None:
            algo = self.ALPHABETA
        if heur == None:
            heur = self.E1
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
            self.currentD1 = 0
            self.currentD2 = 0
            self.currentT = 0.0
            self.miniMaxStartT = 0.0
            self.alphaBetaStartT = 0.0

            if (self.player_turn == 'X' and player_x == self.HUMAN) or (
                    self.player_turn == 'O' and player_o == self.HUMAN):
                (x, y) = self.input_move()
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):

                start = time.time()
                if algo == self.MINIMAX:
                    self.miniMaxStartT = time.time()
                    if self.player_turn == 'X':
                        (_, x, y) = self.minimax(max=False)
                    else:
                        (_, x, y) = self.minimax(max=True)
                else: # algo == self.ALPHABETA:
                    self.alphaBetaStartT = time.time()
                    if self.player_turn == 'X':
                        (m, x, y) = self.alphabeta(max=False, heur=heur)
                    else:
                        (m, x, y) = self.alphabeta(max=True, heur=heur)
                end = time.time()

                print("Maximum D1: " + str(self.currentD1) + " || Maximum D2: " + str(self.currentD2))
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()


def main():
    # bboard=[[0, 0], [1, 1], [2, 2], [3, 3]]
    g = Game(n=3, b=0, s=3, d1=300000, d2=300000, t=20)
    # g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
    # g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)
    # g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.AI)

    g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI, heur=Game.E1)


if __name__ == "__main__":
    main()
