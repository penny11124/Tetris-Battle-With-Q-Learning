from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv


import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import deque
from dqn.deep_q_network import *

from Group37_model import *

import TetrisBattle.tetris as tr
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="my_models")
    # parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args

def test(opt):
    torch.manual_seed(123)
    model38 = torch.load("{}/tetris_10000".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model38.eval()

    model37 = DeepQNetwork()
    env = TetrisDoubleEnv(gridchoice="none", obs_type="image", mode="human")
    env.reset()

    while True:
        if env.game_interface.now_player == 0:
            player=env.game_interface.tetris_list[env.game_interface.now_player]
            tetris = player['tetris']
            com_event = player["com_event"]
            next_steps = env.game_interface.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            predictions38 = model38(next_states)[:, 0]
            index38 = torch.argmax(predictions38).item()
            action = next_actions[index38]

            new_x, new_y = get_pos(tetris)
            drop = False
            for i in action:
                if i == 0:
                    continue
                elif i == 1:
                    #hold the block
                    com_event.set([1])
                    for evt in com_event.get():
                        tetris.trigger(evt)
                elif i == 2:
                    drop = True
                    break
                elif i == 3:
                    tetris.block.rotate()
                elif i == 4:
                    continue
                elif i == 5:
                    new_x += +1
                elif i == 6:
                    new_x += -1
                else:
                    assert False
            tetris.px = new_x
            _, reward, done, infos = env.step(0)
            if drop:
                _, reward, done, infos = env.step(2)
        else:
            tetris = env.game_interface.tetris_list[env.game_interface.now_player]["tetris"]
            next_states = get_next_states(tetris)
            next_actions, next_states = zip(*next_states.items())
            next_states = torch.tensor(next_states,dtype=torch.float)
            predictions37 = model37(next_states)[:, 0]
            index37 = torch.argmax(predictions37).item()
            action = next_actions[index37]
            _, reward, done, infos  = do_action(action, env, tetris)

        env.take_turns()

        if done:
            break
    return infos["winner"]


if __name__ == "__main__":
    opt = get_args()
    winner_is_37 = 0
    winner_is_38 = 0
    num = 20
    for i in range(num):
        winner = 38 - test(opt)
        if  winner == 37:
            winner_is_37 += 1
        else:
            winner_is_38 += 1
        print('\n####################')
        print(f'# {i+1}th fight       #')
        print(f'# WINNER: GROUP {winner} #')
        print('#                  #')
        print('####################')
        
        print(f"37: {winner_is_37} win rate: {winner_is_37/(i+1)*100} %")
        print(f"38: {winner_is_38} win rate: {winner_is_38/(i+1)*100} %")