# -*- coding: utf-8 -*-
import numpy as np

def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You can assume there will always be a solution.
    """
    
    #print(board)
    all_ones = 1
    
    
    mini_size = -1
    for pent in pents:
        pent_size = 0
        for i in range(pent.shape[0]):
            for j in range(pent.shape[1]):
                if pent[i][j] != 0:
                    pent_size += 1
        if (mini_size == -1 or pent_size < mini_size):
            mini_size = pent_size
            
            
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if (board[i][j] == 0):
                all_ones = 0
    if (all_ones == 1):
        new_board = np.zeros(board.shape)
    else:
        new_board = np.copy(board)
        
        
    ret = []
    solved = 0
    if (board.shape[0] >= board.shape[1]):
        for i in range(new_board.shape[0]):
            for j in range(new_board.shape[1]):
                if (solved == 1):
                    break
                if (new_board[i][j] != 0):
                    continue
                for pentIdx in range(len(pents)):
                    curr_pents = []
                    curr_pents.extend(pents)
                    curr_pent = curr_pents.pop(pentIdx)
                    new_pents = []
                    for flipnum in range(2):
                        if flipnum > 0:
                            curr_pent = np.flip(curr_pent, flipnum-1)
                        for rot_num in range(4):
                            al_in = 0
                            for pent in new_pents:
                                if (curr_pent.shape == pent.shape):
                                    if ((curr_pent == pent).all()):
                                        al_in = 1
                                        break
                            if (al_in == 0):
                                new_pents.extend([curr_pent])
                            curr_pent = np.rot90(curr_pent)
                    for new_pent in new_pents:
                        SR = new_pent.shape[0]
                        SC = new_pent.shape[1]
                        #position of the pents
                        for pr in range(SR):
                            for pc in range(SC):
                                curr_board = np.copy(new_board)
                                conflict = 0
                                #index of blocks
                                for r in range(SR):
                                    if (conflict == 1):
                                        break
                                    for c in range(SC):
                                        if (i + pr + r - SR + 1 >= curr_board.shape[0] or j + pc + c - SC + 1 >= curr_board.shape[1] or i + pr + r - SR + 1 < 0 or j + pc + c - SC + 1 < 0):
                                            if (new_pent[r][c] != 0):
                                                conflict = 1
                                                break
                                            else:
                                                continue
                                        if (curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] != 0 and new_pent[r][c] != 0):
                                            conflict = 1
                                            break
                                        if (curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] == 0 and new_pent[r][c] != 0):
                                            curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] = new_pent[r][c]
                                if (conflict == 0):
                                    
                                    do_skip = 0
                                    for r in range(new_pent.shape[0] + 2):
                                        if (do_skip == 1):
                                            break
                                        for c in range(new_pent.shape[1] + 2):
                                            if (i + pr - SR + r >= 0 and i + pr - SR + r < curr_board.shape[0] and j + pc - SC + c >= 0 and j + pc - SC + c < curr_board.shape[1]):
                                                if curr_board[i + pr - SR + r][j + pc - SC + c] == 0:
                                                    size_check = block_check(curr_board, [[i + pr - SR + r, j + pc - SC + c]], mini_size)
                                                    if (size_check == False):
                                                        do_skip = 1
                                                    
                                    if (do_skip == 1):
                                        break
                                    
                                    solution = []
                                    if (len(curr_pents) != 0):
                                        solution = solve(curr_board, curr_pents)
                                    if (len(solution) == len(curr_pents)):
                                        ret.extend(solution)
                                        ret.insert(0, (new_pent, (i + pr  - SR + 1, j + pc  - SC + 1)))
                                        solved = 1
                                        return ret
                return ret
    else:
        for j in range(new_board.shape[1]):
            for i in range(new_board.shape[0]):
                if (solved == 1):
                    break
                if (new_board[i][j] != 0):
                    continue
                for pentIdx in range(len(pents)):
                    curr_pents = []
                    curr_pents.extend(pents)
                    curr_pent = curr_pents.pop(pentIdx)
                    new_pents = []
                    for flipnum in range(2):
                        if flipnum > 0:
                            curr_pent = np.flip(curr_pent, flipnum-1)
                        for rot_num in range(4):
                            al_in = 0
                            for pent in new_pents:
                                if (curr_pent.shape == pent.shape):
                                    if ((curr_pent == pent).all()):
                                        al_in = 1
                                        break
                            if (al_in == 0):
                                new_pents.extend([curr_pent])
                            curr_pent = np.rot90(curr_pent)
                    for new_pent in new_pents:
                        SR = new_pent.shape[0]
                        SC = new_pent.shape[1]
                        #position of the pents
                        for pr in range(SR):
                            for pc in range(SC):
                                curr_board = np.copy(new_board)
                                conflict = 0
                                #index of blocks
                                for r in range(SR):
                                    if (conflict == 1):
                                        break
                                    for c in range(SC):
                                        if (i + pr + r - SR + 1 >= curr_board.shape[0] or j + pc + c - SC + 1 >= curr_board.shape[1] or i + pr + r - SR + 1 < 0 or j + pc + c - SC + 1 < 0):
                                            if (new_pent[r][c] != 0):
                                                conflict = 1
                                                break
                                            else:
                                                continue
                                        if (curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] != 0 and new_pent[r][c] != 0):
                                            conflict = 1
                                            break
                                        if (curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] == 0 and new_pent[r][c] != 0):
                                            curr_board[i + pr + r - SR + 1][j + pc + c - SC + 1] = new_pent[r][c]
                                if (conflict == 0):
                                    
                                    do_skip = 0
                                    for r in range(new_pent.shape[0] + 2):
                                        if (do_skip == 1):
                                            break
                                        for c in range(new_pent.shape[1] + 2):
                                            if (i + pr - SR + r >= 0 and i + pr - SR + r < curr_board.shape[0] and j + pc - SC + c >= 0 and j + pc - SC + c < curr_board.shape[1]):
                                                if curr_board[i + pr - SR + r][j + pc - SC + c] == 0:
                                                    size_check = block_check(curr_board, [[i + pr - SR + r, j + pc - SC + c]], mini_size)
                                                    if (size_check == False):
                                                        do_skip = 1
                                                    
                                    if (do_skip == 1):
                                        break
                                    
                                    solution = []
                                    if (len(curr_pents) != 0):
                                        solution = solve(curr_board, curr_pents)
                                    if (len(solution) == len(curr_pents)):
                                        ret.extend(solution)
                                        ret.insert(0, (new_pent, (i + pr  - SR + 1, j + pc  - SC + 1)))
                                        solved = 1
                                        return ret
                return ret
    return ret




def path_check (board, point):
    path = []
    for k in range(3):
        for h in range(3):
            if ((k != 1 and h == 1) or (k == 1 and h != 1)):
                if (point[0] + k - 1 >= 0 and point[0] + k - 1 < board.shape[0] and point[1] + h - 1 >= 0 and point[1] + h - 1 < board.shape[1]):
                    if (board[point[0] + k - 1][point[1] + h - 1] == 0):
                        path.append([point[0] + k - 1, point[1] + h - 1])
    return path
    
def block_check(board, points, mini_size):
    if (len(points) == mini_size):
        return True
    new_points = []
    new_points.extend(points)
    for point in points:
        path = path_check(board, point)
        for next_p in path:
            if next_p not in new_points:
                new_points.append(next_p)
                if (len(new_points) == mini_size):
                    return True
    if (len(new_points) == len(points)):
        return False
    if (block_check(board, new_points, mini_size)):
        return True
    return False
        
    
    
    
    
    
    