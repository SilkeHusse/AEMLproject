cols = 10
rows = 20

import numpy as np

def get_state_t(board):
    board = np.array(board)
    current_state = [0] * len(board[0])
    for i in range(len(board[0])):
        col = board[:-1,i]
        col = col[::-1]
        print(col)
        max_item = 0
        for idx in range(len(col)):
            if col[idx] != 0:
                max_item = idx + 1
                #current_state[i] = [i,item]
        #for j in range(len(board)-1,-1,-1):
        current_state[i] = max_item
    return current_state

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1 ][cx+off_x] += val
    return mat1

def new_board():
    board = [
        [ 0 for x in range(cols) ]
        for y in range(rows)
    ]
    board += [[ 1 for x in range(cols)]] ### for what?? (is necessary!)
    return board

board = join_matrixes(new_board() , [[4,0,0],[1,1,1],[0,1,0]], (3,18))
print(board)
print(get_state_t(board))