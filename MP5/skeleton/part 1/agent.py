import numpy as np
import utils
import random
from math import floor


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.s = None
        self.a = None

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        # reansfer state
        # state is (adjoining_wall_y, adjoining_wall_x, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        next_state = self.state_transfer(state)

        # update Q and N 
        if (self._train):
            self.upgrade_Q(next_state, points, dead)
        
        if dead:
            # print(len(self.actions))
            self.reset()
            return None

        # next step decision
        a_p, max_v = self.exploration_function(next_state)
            
        if (self._train):
            s1, s0, s2, s3, s4, s5, s6, s7 = next_state
            self.N[s0][s1][s2][s3][s4][s5][s6][s7][a_p] += 1
        

        # update s, a and points
        self.s = next_state
        self.a = a_p
        self.points = points

        # put new value into actions and return
        # self.actions.insert(0, a_p)
        return a_p

    def exploration_function(self, state):
        # state is (adjoining_wall_y, adjoining_wall_x, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        s1, s0, s2, s3, s4, s5, s6, s7 = state
        Us = [-np.inf, -np.inf, -np.inf, -np.inf]
        for i in [4, 5, 6, 7]:
            # if (state[i] == 0 and state[floor((i - 4)/2)] != (i - 4) % 2 + 1):
            if (self.N[s0][s1][s2][s3][s4][s5][s6][s7][i - 4] < self.Ne):
                Us[i - 4] = 1
            else:
                Us[i - 4] = self.Q[s0][s1][s2][s3][s4][s5][s6][s7][i - 4]
        agr_max = 0
        max_v = -np.inf
        for i in range(4):
            if (Us[i] >= max_v):
                agr_max = i
                max_v = Us[i]
        return agr_max, max_v

    def state_transfer(self, state):
        # set adjoining_wall
        adjoining_wall_x = 0
        if (state[0] == utils.GRID_SIZE):
            adjoining_wall_x = 1
        elif (state[0] == utils.DISPLAY_SIZE - utils.GRID_SIZE * 2):
            adjoining_wall_x = 2

        adjoining_wall_y = 0
        if (state[1] == utils.GRID_SIZE):
            adjoining_wall_y = 1
        elif (state[1] == utils.DISPLAY_SIZE - utils.GRID_SIZE * 2):
            adjoining_wall_y = 2

        # set food_dir
        food_dir_x = 0
        if (state[0] > state[3]):
            food_dir_x = 1
        elif (state[0] < state[3]):
            food_dir_x = 2

        food_dir_y = 0
        if (state[1] > state[4]):
            food_dir_y = 1
        elif (state[1] < state[4]):
            food_dir_y = 2

        # set adjoining_body
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0

        for seg in state[2]:
            if (seg[0] == state[0] and seg[1] == state[1] - utils.GRID_SIZE):
                adjoining_body_top = 1
            if (seg[0] == state[0] and seg[1] == state[1] + utils.GRID_SIZE):
                adjoining_body_bottom = 1
            if (seg[0] == state[0] - utils.GRID_SIZE and seg[1] == state[1]):
                adjoining_body_left = 1
            if (seg[0] == state[0] + utils.GRID_SIZE and seg[1] == state[1]):
                adjoining_body_right = 1

        return (adjoining_wall_y, adjoining_wall_x, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)

    def upgrade_Q(self, next_state, points, dead):
        if self.s == None:
            return

        s1, s0, s2, s3, s4, s5, s6, s7 = self.s
        n1, n0, n2, n3, n4, n5, n6, n7 = next_state

        reward = -0.1
        if (points > self.points):
            reward = 1
        elif (dead):
            reward = -1

        alpha = self.C / (self.C + self.N[s0][s1][s2][s3][s4][s5][s6][s7][self.a])

        Q_max = -np.inf
        for dira in range(4):
            Q_max = max(Q_max, self.Q[n0][n1][n2][n3][n4][n5][n6][n7][dira])

        new_Q = reward + self.gamma * Q_max

        self.Q[s0][s1][s2][s3][s4][s5][s6][s7][self.a] += alpha * (new_Q - self.Q[s0][s1][s2][s3][s4][s5][s6][s7][self.a])
        return