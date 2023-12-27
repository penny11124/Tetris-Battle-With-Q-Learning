from TetrisBattle.envs.tetris_env import TetrisSingleEnv

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

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="my_models")

    args = parser.parse_args()
    return args

def train(opt):
    torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = TetrisSingleEnv(gridchoice="none", obs_type="image", mode="rgb_array") # env: gym environment for Tetris
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    
    state = env.reset()

    replay_memory = deque(maxlen=opt.replay_memory_size)

    action_meaning_table = env.get_action_meanings() # meaning of action number

    epoch = 0
    piece_cnt = 0
    while epoch < opt.num_epochs:
        
        next_steps = env.get_next_states() ## 8*4 matrix
        
        # print(next_steps)
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random() 
        # print(u, epsilon)
        random_action = u <= epsilon
        

        ## next_actions: 8*1 matrix, next_states: 8*4 matrix
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        # print(next_states)
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
            # print("random action: ", index)
        else:
            index = torch.argmax(predictions).item()
 
        ## next_state: 1*5 matrix
        next_state = next_states[index, :]
        # print("next_state: ", next_state)
        action = next_actions[index]
        # print("next_actions: ", next_actions)
        # for i in action:
        #     ob, reward, done, infos = env.step(i)
        piece_cnt += 1

        # print("------------------")
        # print("action_len: ", action_len)
        # print("action: ", action)
        # for index, action_value in enumerate(action):
        #     ob, reward, done, infos = env.step(action_value, index, action_len)
        #     # print("index: ", index, "action_value: ", action_value, "reward: ", reward)
        #     if index == action_len - 1:
        #         temp_reward = reward
        #         temp_done = done


        # action = [0,6,0,6,0,2,]
        # action = [0,5,0,5,0,2,]
        # action = [0,3,0,6,0,6,0,2,]
        # action = [0,3,0,5,0,5,0,2,]
        # action = [0,3,0,3,0,6,0,6,0,2,]
        # action = [0,3,0,3,0,5,0,5,0,2,]
        # action = [0,3,0,3,0,3,0,6,0,6,0,2,]
        # action = [0,3,0,3,0,3,0,5,0,5,0,2,]
    
        # action = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]
        # print("action: ", action)
        for i in range(len(action)):
            ob, reward, done, infos = env.step(action[i])
        print("reward after ", piece_cnt,"th piece: ", reward)
        replay_memory.append([state, reward, next_state, done])
        
        if done:
            # final_score = env.score
            # final_tetrominoes = env.tetrominoes
            # final_cleared_lines = env.cleared_lines
            state = env.reset()
            piece_cnt = 0
        else:
            state = next_state
            continue
        print("replay_memory: ", len(replay_memory), "/", opt.replay_memory_size / 10)
        print("------------------")
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        epoch += 1

        # a_memory = [i for i in replay_memory if i.shape == torch.Size([5])]
        # print(a_memory)
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        q_values = model(state_batch)
        # print(q_values)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: " + str(epoch) + "/" + str(opt.num_epochs));

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))

    # =====================================================================================================
    
if __name__ == "__main__":
    opt = get_args()
    train(opt)
    
    
    

