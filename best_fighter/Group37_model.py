from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from TetrisBattle.tetris import Tetris, collideDown
from TetrisBattle.settings import *
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from TetrisBattle.getState import *

def get_pos(tetris):
    return tetris.px, tetris.py
def get_infos(board):
    # board is equal to grid

    # Initialize some stuff
    heights = [0] * len(board)
    diffs = [0] * (len(board) - 1)
    holes = 0
    diff_sum = 0
    cleared = 0
    # Calculate the maximum height of each column
    for i in range(0, len(board)):  # Select a column
        for j in range(0, len(board[0])):  # Search down starting from the top of the board
            if int(board[i][j]) > 0:  # Is the cell occupied?
                heights[i] = len(board[0]) - j  # Store the height value
                break
    
    # Calculate the difference in heights
    for i in range(0, len(diffs)):
        diffs[i] = heights[i + 1] - heights[i]

    # Calculate the cleared lines
    for j in range(0, len(board[0])):
        temp_sum = 0
        for i in range(0, len(board)):  # Select a column
            if int(board[i][j]) > 0:  # Is the cell occupied?
                temp_sum += 1
        cleared += int(temp_sum == GRID_WIDTH)

    # Count the number of holes
    for i in range(0, len(board)):
        occupied = 0  # Set the 'Occupied' flag to 0 for each new column
        for j in range(0, len(board[0])):  # Scan from top to bottom
            if int(board[i][j]) > 0:
                occupied = 1  # If a block is found, set the 'Occupied' flag to 1
            if int(board[i][j]) == 0 and occupied == 1:
                holes += 1  # If a hole is found, add one to the count

    height_sum = sum(heights)

    for i in diffs:
        diff_sum += abs(i)
    return cleared, holes, diff_sum ,height_sum

def get_board(tetris):
    excess = len(tetris.grid[0]) - GRID_DEPTH
    return_grids = np.zeros(shape=(GRID_WIDTH, GRID_DEPTH), dtype=np.float32)
    
    block, px, py = tetris.block, tetris.px, tetris.py
    excess = len(tetris.grid[0]) - GRID_DEPTH
    b = block.now_block()

    for i in range(len(tetris.grid)):
        return_grids[i] = np.array(tetris.grid[i][excess:GRID_DEPTH+excess], dtype=np.float32)
    return_grids[return_grids > 0] = 1

    add_y = hardDrop(tetris.grid, tetris.block, tetris.px, tetris.py)

    for x in range(BLOCK_WIDTH):
        for y in range(BLOCK_LENGTH):
            if b[x][y] > 0:
                # draw ghost grid
                if -1 < px + x < 10 and -1 < py + y + add_y - excess < 20:
                    return_grids[px + x][py + y + add_y - excess] = 1
                else :
                    return None
    return return_grids
def hardDrop(grid, block, px, py):
    y = 0
    x = 0
    if collideDown(grid, block, px, py) == False:
        x = 1
    if x == 1:
        while True:
            py += 1
            y += 1
            if collideDown(grid, block, px, py) == True:
                break
    return y

def get_next_states(tetris):
    states = {}
    org_px, org_py = get_pos(tetris)
    piece_id = tetris.block.block_type()
    if piece_id == 'O':  # O piece
        num_rotations = 1
    elif piece_id == 'S' or piece_id == 'I' or piece_id == 'Z':
        num_rotations = 2
    else:
        num_rotations = 4
    for i in range(num_rotations):
        tetris.block.rotate()
        tetris.py = 0
        for x in range(-2, 10, 1):
            tetris.px = x
            tetris.py = hardDrop(tetris.grid, tetris.block, x, 0)
            board = get_board(tetris)
            if board is not None:
                states[(x, tetris.block.current_shape_id)] = get_infos(board)
    tetris.px, tetris.py = org_px, org_py
    return states
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self.load_state_dict(torch.load('tetris'))

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
def do_action(action, env, tetris):
    #ret_done = False
    tetris.block.current_shape_id = action[1]
    tetris.px = action[0]
    _, reward, done, infos = env.step(0)
    #print(infos)
    #assert(tetris.px == action[0])
    #while infos['is_fallen'] == 0:
    _, reward, done, infos = env.step(2)
    return _, reward, done, infos