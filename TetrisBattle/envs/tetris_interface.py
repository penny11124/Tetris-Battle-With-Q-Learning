import os
import abc
import numpy as np
import random
import torch
import copy
import time

from TetrisBattle.settings import *

from TetrisBattle.tetris import *

from TetrisBattle.renderer import Renderer


POS_LIST = [
    {
        'combo': (44, 437),
        'tetris': (314, 477),
        'tspin': (304, 477),
        'back2back': (314, 437),
        'board': (112, 138),
        'drawscreen': (112, 138),
        'big_ko': (44, 235),
        'ko': (140, 233),
        'transparent': (110, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (298, 140, 3, 360),
        'attack_alarm': (298, 481, 3, 18)
    },
    {
        'combo': (415, 437),
        'tetris': (685, 477),
        'tspin': (675, 477),
        'back2back': (685, 437),
        'board': (495, 138),
        'drawscreen': (495, 138),
        'big_ko': (426, 235),
        'ko': (527, 233),
        'transparent': (494, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (680, 140, 3, 360),
        'attack_alarm': (680, 481, 3, 18)
    }
]

class ComEvent:
    '''
    IO for the AI-agent, which is simulated the pygame.event
    '''
    def __init__(self):
        self._pre_evt_list = []
        self._now_evt_list = []

    def get(self):
        return self._now_evt_list

    def set(self, actions):
        # action: list of int

        self._now_evt_list = []

        for evt in self._pre_evt_list:
            if evt.type == pygame.KEYDOWN or evt.type == "HOLD":
                if evt.key not in actions:
                # if evt.key != action:
                    self._now_evt_list.append(ComEvt(pygame.KEYUP, evt.key))

        for action in actions:
            hold = 0
            for evt in self._pre_evt_list:
                if evt.key == action:
                    if evt.type == pygame.KEYDOWN or evt.type == "HOLD":
                        hold = 1
                        self._now_evt_list.append(ComEvt("HOLD", action))
            if not hold:
                self._now_evt_list.append(ComEvt(pygame.KEYDOWN, action))

        self._pre_evt_list = self._now_evt_list

    def reset(self):
        del self._pre_evt_list[:]
        del self._now_evt_list[:]

class ComEvt:
    '''
    class that score the key informations, it is used in ComEvent
    '''
    def __init__(self, type_, key_):
        self._type = type_
        self._key = key_

    @property
    def key(self):
        return self._key

    @property
    def type(self):
        return self._type

class TetrisInterface(abc.ABC):

    metadata = {'render.modes': ['human', 'rgb_array'], 
                'obs_type': ['image', 'grid']}

    #######################################
    # observation type: 
    # "image" => screen shot of the game 
    # "grid"  => the row data array of the game

    def __init__(self, gridchoice="none", obs_type="image", mode="rgb_array"):

        if mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT)) # SCREEN is 800*600 

        images = load_imgs()

        self.renderer = Renderer(self.screen, images)

        self._obs_type = obs_type

        self._mode = mode

        self.time = MAX_TIME

        self._action_meaning = {
            0: "NOOP",
            1: "hold",
            2: "drop",
            3: "rotate_right",
            4: "rotate_left",
            5: "right",
            6: "left",
            7: "down" 
        }

        self._n_actions = len(self._action_meaning)

        # print(self.action_space.n)

        self._action_set = list(range(self._n_actions))
        
        self.repeat = 1 # emulate the latency of human action

        self.myClock = pygame.time.Clock() # this will be used to set the FPS(frames/s) 

        self.timer2p = pygame.time.Clock() # this will be used for counting down time in our game

        self.tetris_list = []
        self.num_players = -1
        self.now_player = -1

        # whether to fix the speed cross device. Do this by 
        # fix the FPS to FPS (100)
        self._fix_speed_cross_device = True
        self._fix_fps = FPS

    @property 
    def action_meaning(self):
        return self._action_meaning

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def action_set(self):
        return self._action_set

    def screen_size(self):
        # return (x, y)
        return [SCREENHEIGHT, SCREENWIDTH]
    
    def get_screen_shot(self):
        ob = pygame.surfarray.array3d(pygame.display.get_surface())
        ob = np.transpose(ob, (1, 0, 2))
        return ob

    def get_seen_grid(self):
        grid_1 = self.tetris_list[self.now_player]["tetris"].get_grid()
        grid_1[-1][-1] = self.time / MAX_TIME
        # print(grid_1)
        grid_2 = self.tetris_list[1 - self.now_player]["tetris"].get_grid()
        grid_2[-1][-1] = self.time / MAX_TIME
        grid_2.fill(0) # since only one player
        grid = np.concatenate([grid_1, grid_2], axis=1)

        return grid
        # return self.tetris_list[self.now_player]["tetris"].get_grid().reshape(GRID_DEPTH, GRID_WIDTH, 1)

    def get_obs(self):
        if self._obs_type == "grid":
            return self.get_seen_grid()
        elif self._obs_type == "image":
            img = self.get_screen_shot()
        return img

    def random_action(self):
        return random.randint(0, self._n_actions - 1)

    def getCurrentPlayerID(self):
        return self.now_player

    def take_turns(self):
        self.now_player += 1
        self.now_player %= self.num_players
        return self.now_player

    def reward_func(self, infos):
        # define the reward function based on the given infos
        raise NotImplementedError

    def update_time(self, _time):
        # update the time clock and return the running state
        
        if self._fix_speed_cross_device:
            time_per_while = 1 / self._fix_fps * 1000 # transform to milisecond
        else:
            time_per_while = self.timer2p.tick()      # milisecond

        if _time >= 0:                
            _time -= time_per_while * SPEED_UP
        else:
            _time = 0

        return _time

    def task_before_action(self, player):
        # set up the clock and curr_repeat_time
        # set the action to last_action if curr_repeat_time != 0

        self.timer2p.tick() # start calculate the game time
        player["curr_repeat_time"] += 1
        player["curr_repeat_time"] %= self.repeat

    def get_true_action(self, player, action):
        if player["curr_repeat_time"] != 0:
            action = player["last_action"]
        
        player["last_action"] = action

        return action

    def reset(self):
        # Reset the state of the environment to an initial state

        self.time = MAX_TIME
        self.now_player = random.randint(0, self.num_players - 1)
        self.total_reward = 0
        self.curr_repeat_time = 0 # denote the current repeat times
        self.last_infos = {'height_sum': 0, 
                           'diff_sum': 0,
                           'max_height': 0,
                           'holes': 0,
                           'n_used_block': 0,
                           'heights_for_col':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    
        for i, player in enumerate(self.tetris_list):
            if i + 1 > self.num_players:
                break 
            tetris = player["tetris"]
            com_event = player["com_event"]
            pos = player["pos"]
            player["curr_repeatime"] = 0
            player["last_action"] = 0
            tetris.reset()

            com_event.reset()
            self.renderer.drawByName("gamescreen", pos["gamescreen"][0], pos["gamescreen"][1]) # blitting the main background

            self.renderer.drawGameScreen(tetris)

        self.renderer.drawTime2p(self.time)
       
        #time goes until it hits zero
        #when it hits zero return endgame screen
        
        pygame.display.flip()
        self.myClock.tick(FPS)  

        ob = self.get_obs()

        return ob


class TetrisSingleInterface(TetrisInterface):

    metadata = {'render.modes': ['human', 'rgb_array'], 
                'obs_type': ['image', 'grid']}

    #######################################
    # observation type: 
    # "image" => screen shot of the game 
    # "grid"  => the row data array of the game

    def __init__(self, gridchoice="none", obs_type="image", mode="rgb_array"):
        super(TetrisSingleInterface, self).__init__(gridchoice, obs_type, mode)
        self.num_players = 1

        # The second player is dummy, it is used for 
        # self.renderer.drawByName("transparent", *opponent["pos"]["transparent"]) at around line 339
        for i in range(self.num_players + 1):
            info_dict = {"id": i}

            # adding the action information
            for k, v in self._action_meaning.items():
                info_dict[v] = k

            self.tetris_list.append({
                'info_dict': info_dict,
                'tetris': Tetris(Player(info_dict), gridchoice),
                'com_event': ComEvent(),
                'pos': POS_LIST[i],
                'curr_repeat_time': 0,
                'last_action': 0
            })
            
        self.reset()
    
    def reward_func(self, infos):

        if infos['is_fallen']:
            basic_reward = infos['scores']
            # additional_reward = 0.01 if infos['holes'] == 0 else 0

            # additional_reward = -0.51 * infos['height_sum'] + 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            additional_reward = 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            # additional_reward = infos['cleared'] # + (0.2 if infos['holes'] == 0 else 0)
            # return basic_reward + 0.01 * additional_reward - infos['penalty']
            return basic_reward + 1 * additional_reward + infos['reward_notdie']
        
        return 0

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board
    
    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(10)]] + board
        return board
    
    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < 20 and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes
    
    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
        heights = 20 - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        # print([lines_cleared, holes, bumpiness, height])
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_next_states(self):
        states = {}
        # pos_info = {}
        player, opponent = self.tetris_list[self.now_player], self.tetris_list[::-1][self.now_player]
        tetris = player["tetris"]

        # print(cur_piece_id, end=' ')
        def clear_garbage(board):
            board = board.transpose()
            cnt = 0
            for lines in board:
                if sum(lines) == 10:
                    cnt+=1
            if cnt > 0:
                empty_line = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
                import numpy as np
                new_board = np.array([empty_line for i in range(cnt)])
                board = np.concatenate((new_board,tetris.get_board().transpose()),axis=0)[:-5]
            return board.transpose()
        board = clear_garbage(tetris.get_board())
        
        x = tetris.px
        y = tetris.py
        # print(cur_piece_id)

        if tetris.held is not None:
            for hold in range(2):
                if hold:
                    block = copy.copy(tetris.held)
                else:
                    block = copy.copy(tetris.block)
                piece_id = block.block_type()

                for j in range(4):
                    for i in range(10):
                    
                        board = tetris.get_board()
                        next_x = i
                        next_dir = j
                        block.current_shape_id = next_dir
                        
                        new_4 = 0
                        

                        if piece_id == 'I':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - i_len[next_dir]:
                                next_x += i_x[next_dir]
                            else :
                                continue 
                            new_4 = 4 - i_x[j]
            
                        elif piece_id == 'O':
                            # if next_dir == 1 or next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - o_len[next_dir]:
                                next_x += o_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - o_x[j]
                        elif piece_id == 'J':
                            if next_x <= 10 - j_len[next_dir]:
                                next_x += j_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - j_x[j]
                        elif piece_id == 'L':
                            if next_x <= 10 - l_len[next_dir]:
                                next_x += l_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - l_x[j]
                        elif piece_id == 'Z':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - z_len[next_dir]:
                                next_x += z_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - z_x[j]
                        elif piece_id == 'S':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - s_len[next_dir]:
                                next_x += s_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - s_x[j]
                        elif piece_id == 'T':
                            if next_x <= 10 - t_len[next_dir]:
                                next_x += t_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - t_x[j]
                        # print(next_x)
                        newy = hardDrop(board, block, next_x, y)
                        newy = newy - 2
                        put_block_in_grid(board, block, next_x, newy)
                        # print(np.transpose(board))
                        # print('------------------')
                        
                        instruction = [0]
                        
                        if hold:
                            instruction.append(1)
                            instruction.append(0)

                        for c in range(j):
                            instruction.append(3)
                            instruction.append(0)
                        if new_4 > i:
                            # print(4 - i_x[j] - i)
                            for a in range(new_4 - i):
                                instruction.append(6)
                                instruction.append(0)
                        else:
                            # print(i - 4 + i_x[j])
                            for b in range(i - new_4):
                                instruction.append(5)
                                instruction.append(0)

                        instruction.append(2)

                        # print(instruction)
                        
                        # #print the board as a matrix with  as 1 and ⬛ as 0
                        # board_view = np.where(board == 0, '⬛', '⬜')
                        # #print only the last 5 rows
                        # print(np.transpose(board_view)[13:], '\n')

                        board_list = np.transpose(board).tolist()
                        # print(np.transpose(board).tolist())4 - i_x[j]
                        states[tuple(instruction)] = self.get_state_properties(board_list)
                # print(piece_id, instruction, ": ", states[tuple(instruction)])
            # print("----------------------------------")

        else:
            instruction = [0,1]
            board_list = np.transpose(board).tolist()
            # print(np.transpose(board).tolist())4 - i_x[j]
            states[tuple(instruction)] = self.get_state_properties(board_list)
        
        return states
        

    def act(self, action):
        # Execute one time step within the environment
        
        end = 0
        scores = 0

        player, opponent = self.tetris_list[self.now_player], self.tetris_list[::-1][self.now_player]
        tetris = player["tetris"]
        com_event = player["com_event"]
        pos = player["pos"]

        self.task_before_action(player)

        # mapping_dict = {
        #     "NOPE": 0,
        #     pygame.K_c: 1,
        #     pygame.K_SPACE: 2,
        #     pygame.K_UP: 3,
        #     pygame.K_z: 4,
        #     pygame.K_RIGHT: 5,
        #     pygame.K_LEFT: 6,
        #     pygame.K_DOWN: 7
        # }

        # action = 0

        # for evt in pygame.event.get():
        #     if evt.type == pygame.KEYDOWN:
        #         if mapping_dict.get(evt.key) is not None:
        #             action = mapping_dict[evt.key]
        #             print(action)

        action = self.get_true_action(player, action)

        tetris.natural_down()

        com_event.set([action])

        for evt in com_event.get():
            tetris.trigger(evt)

        tetris.move()

        # print(tetris.get_grid()[:20, :10])
        
        scores = 0
        
        penalty_die = 0

        reward_notdie = 0

        if tetris.check_fallen():
            # compute the scores and attack the opponent
            scores = tetris.clear()

            # scores += cleared_scores
            # scores += tetris.cleared

            self.renderer.drawCombo(tetris, pos["combo"][0], pos["combo"][1])

            self.renderer.drawTetris(tetris, pos["tetris"][0], pos["tetris"][1])
            self.renderer.drawTspin(tetris, pos["tspin"][0], pos["tspin"][1])
            self.renderer.drawBack2Back(tetris, pos["back2back"][0], pos["back2back"][1])

            if tetris.check_KO():
                self.renderer.drawBoard(tetris, pos["board"][0], pos["board"][1])
                
                tetris.clear_garbage()

                self.renderer.drawByName("ko", pos["ko"][0], pos["ko"][1])
                self.renderer.drawByName("transparent", pos["transparent"][0], pos["transparent"][1])

                # screen.blit(kos[tetris_2.get_KO() - 1], (426, 235))
                pygame.display.flip()

                # scores -= 5
                penalty_die = self.total_reward * 0.8

                end = 1

            tetris.new_block()

        self.renderer.drawGameScreen(tetris)

        tetris.increment_timer()

        # if tetris.attacked == 0:
        #     pygame.draw.rect(self.screen, (30, 30, 30), pos["attack_clean"]) 

        # if tetris.attacked != 0:
            
        #     for j in range(tetris.attacked):
        #         pos_attack_alarm = list(pos["attack_alarm"])
        #         # modified the y axis of the rectangle, according to the strength of attack
        #         pos_attack_alarm[1] = pos_attack_alarm[1] - 18 * j
        #         pygame.draw.rect(self.screen, (255, 0, 0), pos_attack_alarm) 

        if tetris.KO > 0:
            self.renderer.drawKO(tetris.KO, pos["big_ko"][0], pos["big_ko"][1])
            
        self.renderer.drawScreen(tetris, pos["drawscreen"][0], pos["drawscreen"][1])

        self.renderer.drawByName("transparent", *opponent["pos"]["transparent"])

        self.time = self.update_time(self.time)

        if self.time == 0:
            reward_notdie = 0.3 * self.total_reward
            end = 1

        self.renderer.drawTime2p(self.time)
        
        # time goes until it hits zero
        # when it hits zero return endgame screen
        
        self.myClock.tick(FPS)  
        pygame.display.flip()

        ob = self.get_obs()

        infos = {'is_fallen': tetris.is_fallen}

        if tetris.is_fallen:
            height_sum, diff_sum, max_height, holes, heights_for_col = get_infos(tetris.get_board())

            # store the different of each information due to the move
            infos['height_sum'] = height_sum - self.last_infos['height_sum'] - 4
            infos['diff_sum'] =  diff_sum - self.last_infos['diff_sum']
            infos['max_height'] =  max_height - self.last_infos['max_height']
            infos['holes'] =  holes - self.last_infos['holes'] 
            infos['n_used_block'] =  tetris.n_used_block - self.last_infos['n_used_block']
            infos['is_fallen'] =  tetris.is_fallen 
            infos['scores'] =  scores 
            infos['cleared'] =  tetris.cleared
            infos['penalty'] =  penalty_die
            infos['reward_notdie'] = reward_notdie
            infos['heights_for_col'] = heights_for_col
            # print(tetris.cleared)
            self.last_infos = {'height_sum': height_sum,
                               'diff_sum': diff_sum,
                               'max_height': max_height,
                               'holes': holes,
                               'n_used_block': tetris.n_used_block, 
                               'heights_for_col': heights_for_col}
                               

            # print(infos)

        # reward = self.reward_func(infos)
        # reward = 0

        reward = 1 + ( min(tetris.cleared, 4) ** 2) * 10
        if tetris.is_fallen:
            if max_height > 10:
                reward -= max( max_height - 10, 0)
            if max_height > 15:
                reward -= max( max_height - 15, 0) ** 2 * 5
        if end:
            reward -= 1000
            
        # reward = 1 + tetris.sent * 10 + tetris.cleared
        # if end:
        #     reward -= 1000
        # print("reward after one piece", reward)


        # self.total_reward += reward

        if end:
            # freeze(0.5)
            infos['sent'] = tetris.sent
            # self.reset()

        return ob, reward, end, infos

class TetrisDoubleInterface(TetrisInterface):

    metadata = {'render.modes': ['human', 'rgb_array'], 
                'obs_type': ['image', 'grid']}

    #######################################
    # observation type: 
    # "image" => screen shot of the game 
    # "grid"  => the row data array of the game

    def __init__(self, gridchoice="none", obs_type="image", mode="rgb_array"):
        super(TetrisDoubleInterface, self).__init__(gridchoice, obs_type, mode)
        
        self.num_players = 2

        for i in range(self.num_players):
            info_dict = {"id": i}

            # adding the action information
            for k, v in self._action_meaning.items():
                info_dict[v] = k

            self.tetris_list.append({
                'info_dict': info_dict,
                'tetris': Tetris(Player(info_dict), gridchoice),
                'com_event': ComEvent(),
                'pos': POS_LIST[i],
                'curr_repeat_time': 0,
                'last_action': 0
            })
        self.reset()

    def reward_func(self, infos):

        if infos['is_fallen']:
            basic_reward = infos['scores']
            # additional_reward = 0.01 if infos['holes'] == 0 else 0

            # additional_reward = -0.51 * infos['height_sum'] + 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            additional_reward = 0.76 * infos['cleared'] - 0.36 * infos['holes'] - 0.18 * infos['diff_sum']
            # additional_reward = infos['cleared'] # + (0.2 if infos['holes'] == 0 else 0)
            # return basic_reward + 0.01 * additional_reward - infos['penalty']
            return basic_reward + 1 * additional_reward + infos['reward_notdie']
        
        return 0

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board
    
    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(10)]] + board
        return board
    
    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < 20 and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes
    
    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
        heights = 20 - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        # print([lines_cleared, holes, bumpiness, height])
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
    
    def get_next_states(self):
        states = {}
        # pos_info = {}
        player, opponent = self.tetris_list[self.now_player], self.tetris_list[::-1][self.now_player]
        tetris = player["tetris"]
        garbage = len(tetris.grid[0])-20
        # print(tetris.block.block_type()) 
        temp_board = tetris.get_board()
        # temp_board = np.array(tetris.grid).T
        # temp_board = temp_board[:20]
        # temp_board = temp_board.T
        
        # board_view = np.where(temp_board == 0, '.', '#')
        # print(garbage)
        # print(np.transpose(board_view), '\n')

        

        x = tetris.px
        y = tetris.py

        if tetris.held is not None and self.time > 0:
            for hold in range(2):
                if hold:
                    block = copy.copy(tetris.held)
                else:
                    block = copy.copy(tetris.block)
                piece_id = block.block_type()

                for j in range(4):
                    for i in range(10):
                    
                        # board = np.array(tetris.grid).T
                        # board = board[:20]
                        # board = board.T
                        board = tetris.get_board()
                        for q in range(10):
                            for k in range(20):
                                if k < garbage:
                                    board[q][k] = 0
                                else :
                                    board[q][k] = copy.copy(temp_board[q][k-garbage])            
                        next_x = i
                        next_dir = j
                        block.current_shape_id = next_dir
                        
                        new_4 = 0
                        

                        if piece_id == 'I':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - i_len[next_dir]:
                                next_x += i_x[next_dir]
                            else :
                                continue 
                            new_4 = 4 - i_x[j]
            
                        elif piece_id == 'O':
                            # if next_dir == 1 or next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - o_len[next_dir]:
                                next_x += o_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - o_x[j]
                        elif piece_id == 'J':
                            if next_x <= 10 - j_len[next_dir]:
                                next_x += j_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - j_x[j]
                        elif piece_id == 'L':
                            if next_x <= 10 - l_len[next_dir]:
                                next_x += l_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - l_x[j]
                        elif piece_id == 'Z':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - z_len[next_dir]:
                                next_x += z_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - z_x[j]
                        elif piece_id == 'S':
                            # if next_dir == 2 or next_dir == 3: continue
                            if next_x <= 10 - s_len[next_dir]:
                                next_x += s_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - s_x[j]
                        elif piece_id == 'T':
                            if next_x <= 10 - t_len[next_dir]:
                                next_x += t_x[next_dir]
                            else :
                                continue
                            new_4 = 4 - t_x[j]
                        # print(next_x)
                        newy = hardDrop(board, block, next_x, y)
                        newy = newy - 2 + garbage
                        put_block_in_grid(board, block, next_x, newy)
                        # print(np.transpose(board))
                        # print('------------------')
                        
                        instruction = [0]
                        
                        if hold:
                            instruction.append(1)
                            instruction.append(0)

                        for c in range(j):
                            instruction.append(3)
                            instruction.append(0)
                        if new_4 > i:
                            # print(4 - i_x[j] - i)
                            for a in range(new_4 - i):
                                instruction.append(6)
                                instruction.append(0)
                        else:
                            # print(i - 4 + i_x[j])
                            for b in range(i - new_4):
                                instruction.append(5)
                                instruction.append(0)

                        instruction.append(2)

                        # print(instruction)
                        
                        # #print the board as a matrix with  as 1 and ⬛ as 0
                        # board_view = np.where(board == 0, '⬛', '⬜')
                        # #print only the last 5 rows
                        # print(np.transpose(board_view)[13:], '\n')

                        # board_view = np.where(board == 0, '.', '#')
                        # print(garbage)
                        # print(np.transpose(board_view), '\n')
                        # board_view = np.where(board == 0, '.', '#')
                        # print(garbage)
                        # print(np.transpose(board_view), '\n')

                        board_list = np.transpose(board).tolist()
                        
                        states[tuple(instruction)] = self.get_state_properties(board_list)
                # print(piece_id, instruction, ": ", states[tuple(instruction)])
            # print("----------------------------------")
            
        else:
            instruction = [1]
            board_list = np.transpose(temp_board).tolist()
            # print(np.transpose(temp_board))
            # print(np.transpose(board).tolist())4 - i_x[j]
            states[tuple(instruction)] = self.get_state_properties(board_list)
        
        return states
    def act(self, action):
        # Execute one time step within the environment
        
        end = 0
        scores = 0

        player, opponent = self.tetris_list[self.now_player], self.tetris_list[::-1][self.now_player]
        tetris = player["tetris"]
        com_event = player["com_event"]
        pos = player["pos"]

        self.task_before_action(player)

        action = self.get_true_action(player, action)

        tetris.natural_down()

        com_event.set([action])

        for evt in com_event.get():
            tetris.trigger(evt)

        tetris.move()

        scores = 0

        if tetris.check_fallen():
            # compute the scores and attack the opponent
            scores = tetris.clear()
            opponent["tetris"].add_attacked(scores)

            self.renderer.drawCombo(tetris, *pos["combo"])

            self.renderer.drawTetris(tetris, *pos["tetris"])
            self.renderer.drawTspin(tetris, *pos["tspin"])
            self.renderer.drawBack2Back(tetris, *pos["back2back"])

            if tetris.check_KO():
                
                self.renderer.drawBoard(tetris, *pos["board"])
                
                opponent["tetris"].update_ko()

                tetris.clear_garbage()
                self.renderer.drawByName("ko", *pos["ko"])
                self.renderer.drawByName("transparent", *pos["transparent"])

                # screen.blit(kos[tetris_2.get_KO() - 1], (426, 235))
                pygame.display.flip()

                # scores -= 1

                # end = 1

            tetris.new_block()

        self.renderer.drawGameScreen(tetris)

        tetris.increment_timer()

        if tetris.attacked == 0:
            pygame.draw.rect(self.screen, (30, 30, 30), pos["attack_clean"]) 

        if tetris.attacked != 0:
            
            for j in range(tetris.attacked):
                pos_attack_alarm = list(pos["attack_alarm"])
                # modified the y axis of the rectangle, according to the strength of attack
                pos_attack_alarm[1] = pos_attack_alarm[1] - 18 * j
                pygame.draw.rect(self.screen, (255, 0, 0), pos_attack_alarm) 

        if tetris.KO > 0:
            self.renderer.drawKO(tetris.KO, *pos["big_ko"])
            
        self.renderer.drawScreen(tetris, *pos["drawscreen"])

            # SCREEN.blit(IMAGES["transparent"], (494, 135))

        if Judge.check_ko_win(tetris, max_ko=3):
            end = 1
            winner = tetris.get_id()

        if Judge.check_ko_win(opponent["tetris"], max_ko=3):
            end = 1
            winner = opponent["tetris"].get_id()

        self.time = self.update_time(self.time)

        if self.time == 0:
            winner = Judge.who_win(tetris, opponent["tetris"])
            end = 1

        if self.time > 0: self.renderer.drawTime2p(self.time)
        
        self.myClock.tick(FPS)  
        pygame.display.flip()

        ob = self.get_obs()

        infos = {'now_player': self.now_player}

        if end:
            # print(winner)
            time.sleep(1)
            infos['winner'] = winner

            # self.reset()

        reward = 0#self.reward_func(infos)

        return ob, reward, end, infos
