from random import randrange as rand
import pygame, sys
import numpy as np
import math
from copy import deepcopy

### CONFIGURATIONS ###

cell_size = 25
cols = 10
rows = 20
maxfps = 25

colors = [(0, 0, 0),  # color for background
          (255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (0, 255, 255),
          (128, 0, 128),
          (255, 165, 0),
          (0, 0, 0)]  # helper color for background (grid)

tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]    ]


def rotate_clockwise(shape):
    return [
        [shape[y][x] for y in range(len(shape))]
        for x in range(len(shape[0]) - 1, -1, -1)
    ]
def rotate_anticlockwise(shape):
    return [
        [shape[y][x] for y in range(len(shape) - 1, -1, -1)]
        for x in range(len(shape[0]))
    ]
def rotate_half(shape):
    return rotate_clockwise(rotate_clockwise(shape))

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board  # adds new line on top (highest row)

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1

def new_board():
    board = [
        [0 for x in range(cols)]
        for y in range(rows)
    ]
    board += [[1 for x in range(cols)]]  # usefulness is unclear, but necessary
    return board


def get_state_t(board, x_pos):
    board = np.array(board)

    current_state = [0] * len(board[0])
    for i in range(len(board[0])):
        col = board[:-1, i]
        col = col[::-1]
        max_item = 0
        for idx in range(len(col)):
            if col[idx] != 0:
                max_item = idx + 1
        current_state[i] = max_item

    height = sum(current_state)

    max_height = max(current_state)
    diff_heights = [max_height - current_state[i] for i in range(len(current_state))]
    tower = sum(diff_heights)

    holes = 0
    for k in range(len(board[0])):
        col = board[:-1, k]
        col = col[::-1]
        for y in range(current_state[k]):
            if col[y] == 0:
                holes += 1

    bumpiness = 0
    for j in range(len(current_state) - 1):
        bumpiness += math.fabs(current_state[j] - current_state[j + 1])

    x_pos = x_pos / 9  # normalization
    min_height = min(current_state)
    current_state = [x - min_height for x in current_state]
    level = sum(x == 0 for x in current_state)
    current_state = [x / 20 for x in current_state]  # normalization
    current_state = current_state + [min_height / 20] + [x_pos]

    return current_state, height, bumpiness, holes, tower, level

class TetrisApp(object):

    def __init__(self):
        self.next_stone_idx = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_idx]
        self.next_tetromino = [0] * len(tetris_shapes)
        self.next_tetromino[self.next_stone_idx] = 1
        self.init_game()

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.tetromino = self.next_tetromino[:]
        self.next_stone_idx = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.next_stone_idx]
        self.next_tetromino = [0] * len(tetris_shapes)
        self.next_tetromino[self.next_stone_idx] = 1
        self.stone_x = int(cols / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0
        self.rotation = [1, 0, 0, 0]
        self.pieces += 1

        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.initialization = True
        self.clock = clock = pygame.time.Clock()
        self.actions = 0
        self.pieces = 0
        self.board = new_board()
        self.new_stone()
        self.score = 0
        self.score_old = 0
        self.lines = 0
        self.gameover = False
        self.state, self.height, self.bumpiness, self.holes, _, _ = get_state_t(self.board, self.stone_x)

    def quit(self):
        sys.exit()

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)), (x, y))
            y += 20
    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False, (255, 255, 255), (0, 0, 0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2
            self.screen.blit(msg_image, (self.width // 2 - msgim_center_x, self.height // 2 - msgim_center_y + i * 22))
    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(self.screen, colors[val],
                                     pygame.Rect((off_x + x) * cell_size, (off_y + y) * cell_size, cell_size,
                                                 cell_size), 0)

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score_old = self.score
        self.score += linescores[n]

    def move(self, delta_x):
        if not self.gameover:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
            if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
                self.stone_x = new_x

    def drop(self, screen_update):
        if not self.gameover:
            self.stone_y += 1
            if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
                self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            if screen_update:
                                self.screen.fill(colors[0])
                                self.draw_matrix(self.board, (0, 0))
                                self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
                                self.clock.tick(maxfps)
                                pygame.display.update()
                            self.board = remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False
    def insta_drop(self, screen_bool):
        if not self.gameover:
            while (not self.drop(screen_update=screen_bool)):
                pass

    def rotate_stone(self, direction):
        if not self.gameover:
            current_rot = np.where(np.array(self.rotation) != 0)[0].item()
            if direction == 'clock':
                new_stone = rotate_clockwise(self.stone)
                if current_rot == 0:
                    self.rotation_help = [0, 1, 0, 0]
                elif current_rot == 1:
                    self.rotation_help = [0, 0, 0, 1]
                elif current_rot == 2:
                    self.rotation_help = [1, 0, 0, 0]
                elif current_rot == 3:
                    self.rotation_help = [0, 0, 1, 0]
            elif direction == 'anticlock':
                new_stone = rotate_anticlockwise(self.stone)
                if current_rot == 0:
                    self.rotation_help = [0, 0, 1, 0]
                elif current_rot == 1:
                    self.rotation_help = [1, 0, 0, 0]
                elif current_rot == 2:
                    self.rotation_help = [0, 0, 0, 1]
                elif current_rot == 3:
                    self.rotation_help = [0, 1, 0, 0]
            elif direction == 'half':
                new_stone = rotate_half(self.stone)
                if current_rot == 0:
                    self.rotation_help = [0, 0, 0, 1]
                elif current_rot == 1:
                    self.rotation_help = [0, 0, 1, 0]
                elif current_rot == 2:
                    self.rotation_help = [0, 1, 0, 0]
                elif current_rot == 3:
                    self.rotation_help = [1, 0, 0, 0]

            if not check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
                self.stone = new_stone
                self.rotation = self.rotation_help

    def get_perf(self):
        return self.score, self.lines, self.pieces, self.actions

    def get_next_states(self):
        states = []
        curr_piece = [row[:] for row in self.stone]
        curr_piece_help = curr_piece
        current_board  = deepcopy(self.board)
        current_x = deepcopy(self.stone_x)
        current_stone = deepcopy(self.stone)
        current_rot = deepcopy(self.rotation)
        current_tetromino = deepcopy(self.tetromino)

        current_piece_idx = np.where(np.array(current_tetromino) != 0)[0].item()
        if current_piece_idx in [0,3,4]:
            rot_range = 4
        elif current_piece_idx in [1,2,5]:
            rot_range = 2
        else:
            rot_range = 1

        for rot in range(rot_range):
            if rot == 1:
                curr_piece_help = rotate_clockwise(curr_piece)
            if rot == 2:
                curr_piece_help = rotate_anticlockwise(curr_piece)
            if rot == 3:
                curr_piece_help = rotate_half(curr_piece)

            valid_xs = cols - len(curr_piece_help[0])
            for x in range(valid_xs + 1): # for each position
                if rot == 1:
                    self.rotate_stone("clock")
                if rot == 2:
                    self.rotate_stone("anticlock")
                if rot == 3:
                    self.rotate_stone("half")

                self.stone_x = x
                self.insta_drop(screen_bool=False)
                state,_,_,_,_,_ = get_state_t(self.board, self.stone_x)
                states.append(state)
                self.board = deepcopy(current_board)
                self.stone_x = deepcopy(current_x)
                self.stone = deepcopy(current_stone)
                self.rotation = deepcopy(current_rot)
                self.tetromino = deepcopy(current_tetromino)
        self.pieces -= len(states)

        return states

    def step_grouped(self, action_number, screen):
        # initialize screen for first time

        update_screen = False
        if screen and self.initialization:
            pygame.init()
            pygame.key.set_repeat()  # held keys are not repeated
            self.width = cell_size * (cols)
            self.height = cell_size * rows
            self.rlim = cell_size * cols
            self.bground_grid = [[8 if x % 2 == y % 2 else 0 for x in range(cols)] for y in range(rows)]
            self.default_font = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("TETRIS")
            pygame.event.set_blocked(pygame.MOUSEMOTION)  # block mouse movements
            self.initialization = False

        if screen:
            self.screen.fill(colors[0])
            self.draw_matrix(self.board, (0, 0))
            self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
            pygame.display.update()
            self.clock.tick(maxfps)
            update_screen = True

        #self.gameover = False
        #self.move_penalty = 0
        height_old = self.height
        bumpiness_old = self.bumpiness
        holes_old = self.holes

        current_piece_idx = np.where(np.array(self.tetromino) != 0)[0].item()
        if current_piece_idx in [0, 3, 4]:
            rot_normal = [0, 1, 2, 3, 4, 5, 6, 7]
            rot_clock = [8, 9, 10, 11, 12, 13, 14, 15, 16]
            rot_anti = [17, 18, 19, 20, 21, 22, 23, 24, 25]
            rot_half = [26, 27, 28, 29, 30, 31, 32, 33]

            if action_number in rot_normal:
                position = action_number
            elif action_number in rot_clock:
                self.rotate_stone("clock")
                position = action_number - min(rot_clock)
            elif action_number in rot_anti:
                self.rotate_stone("anticlock")
                position = action_number - min(rot_anti)
            elif action_number in rot_half:
                self.rotate_stone("half")
                position = action_number - min(rot_half)

        elif current_piece_idx in [1, 2]:
            rot_normal = [0, 1, 2, 3, 4, 5, 6, 7]
            rot_rotate = [8, 9, 10, 11, 12, 13, 14, 15, 16]

            if action_number in rot_normal:
                position = action_number
            elif action_number in rot_rotate:
                self.rotate_stone("clock")
                position = action_number - min(rot_rotate)

        elif current_piece_idx == 5:
            rot_normal = [0, 1, 2, 3, 4, 5, 6]
            rot_rotate = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

            if action_number in rot_normal:
                position = action_number
            elif action_number in rot_rotate:
                self.rotate_stone("clock")
                position = action_number - min(rot_rotate)

        elif current_piece_idx == 6:
            position = action_number

        if not check_collision(self.board, self.stone, (position, self.stone_y)):
            self.stone_x = position
            self.insta_drop(update_screen)
            self.actions += 1
        else:
            self.gameover = True

        self.state, self.height, self.bumpiness, self.holes, _, _ = get_state_t(self.board, self.stone_x)

        lines = self.score - self.score_old  # cleared lines reward
        height = self.height - height_old  # accumulative height of cols penalty
        bumpiness = self.bumpiness - bumpiness_old  # adjacent cols difference in height penalty
        holes = self.holes - holes_old  # creation of holes penalty
        #tower = self.tower - tower_old  # free space below max col penalty
        #level = self.level - level_old  # free cells in lowest line reward

        reward = lines/2 - 0.01 * height - 0.075 * bumpiness - 0.05 * holes #- 0.5 * max(tower, 0) - 0.2 * level

        if self.gameover:  # game over penalty
            reward -= 5

        return reward, self.gameover

    def step(self, action_number, screen):
        key_actions = {
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            #'DOWN': lambda: self.rotate_stone('clock'),
            #'UP': lambda: self.rotate_stone('anticlock'),
            #'SPACE': lambda: self.rotate_stone('half'),
            'RETURN': self.insta_drop
        }

        # initialize screen for first time
        if screen and self.initialization:
            pygame.init()
            pygame.key.set_repeat()  # held keys are not repeated
            self.width = cell_size * (cols)
            self.height = cell_size * rows
            self.rlim = cell_size * cols
            self.bground_grid = [[8 if x % 2 == y % 2 else 0 for x in range(cols)] for y in range(rows)]
            self.default_font = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("TETRIS")
            pygame.event.set_blocked(pygame.MOUSEMOTION)  # block mouse movements
            self.initialization = False

        if screen:
            self.screen.fill(colors[0])
            self.draw_matrix(self.board, (0, 0))
            self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
            pygame.display.update()

        self.gameover = False
        self.move_penalty = 0
        height_old = self.height
        bumpiness_old = self.bumpiness
        holes_old = self.holes
        tower_old = self.tower
        level_old = self.level

        self.actions += 1
        key = list(key_actions)[action_number]
        key_actions[key]()

        self.state, self.height, self.bumpiness, self.holes, self.tower, self.level = get_state_t(self.board, self.stone_x)

        height = self.height - height_old # accumulative height of cols penalty
        bumpiness = self.bumpiness - bumpiness_old # adjacent cols difference in height penalty
        holes = self.holes - holes_old # creation of holes penalty
        tower = self.tower - tower_old # free space below max col penalty
        lines = self.score - self.score_old  # cleared lines reward
        level = self.level - level_old # free cells in lowest line reward

        reward = lines - 0.01 * height - 0.5 * bumpiness - 0.5 * holes - 1 * max(tower,0) - 0.2 * level
        reward = reward - self.move_penalty  # collision penalty
        if action_number == 2:  # hard drop reward
            reward += 10
        if self.gameover:  # game over penalty
            reward -= 50

        return reward, self.gameover
