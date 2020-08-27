from random import randrange as rand
import random
import pygame, sys

### CONFIGURATIONS ###

cell_size = 25
cols = 10
rows = 20
maxfps = 15

colors = [  (0, 0, 0), # color for background
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (128, 0, 128),
            (255, 165, 0),
            (0, 0, 0) ] # helper color for background (grid)

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
        [ shape[y][x] for y in range(len(shape)) ]
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
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board # adds new line on top (highest row)

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
    board += [[ 1 for x in range(cols)]] # usefulness is unclear, but necessary
    return board


class TetrisApp(object):

    def __init__(self):
        self.tetromino_idx = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.tetromino_idx]
        self.init_game()

    def new_stone(self):
        self.actions_list.append(self.actions)
        self.actions = 0
        self.stone = self.next_stone[:]
        self.tetromino_idx = rand(len(tetris_shapes))
        self.next_stone = tetris_shapes[self.tetromino_idx]
        self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0
        self.pieces += 1

        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.actions_list = []
        self.actions = 0
        self.pieces = 0
        self.board = new_board()
        self.new_stone()
        self.score = 0
        self.score_old = 0
        self.lines = 0

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit( self.default_font.render(line, False, (255,255,255), (0,0,0)), (x,y))
            y += 20
    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False, (255,255,255), (0,0,0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2
            self.screen.blit(msg_image, (self.width // 2 - msgim_center_x, self.height // 2 - msgim_center_y + i * 22))
    def draw_matrix(self, matrix, offset):
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(self.screen, colors[val],
                        pygame.Rect((off_x+x) * cell_size, (off_y+y) * cell_size, cell_size, cell_size), 0)

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

    def drop(self, manual):
        if not self.gameover:
            self.stone_y += 1
            if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
                self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False
    def insta_drop(self):
        if not self.gameover:
            while(not self.drop(True)):
                pass

    def rotate_stone(self, direction):
        if not self.gameover:
            if direction == 'clock':
                new_stone = rotate_clockwise(self.stone)
            elif direction == 'anticlock':
                new_stone = rotate_anticlockwise(self.stone)
            elif direction == 'half':
                new_stone = rotate_half(self.stone)

            if not check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def step(self, action_number):
        key_actions = {
            'LEFT':     lambda: self.move(-1),
            'RIGHT':    lambda: self.move(+1),
            'DOWN':     lambda: self.rotate_stone('clock'),
            'UP':       lambda: self.rotate_stone('anticlock'),
            'SPACE':    lambda: self.rotate_stone('half'),
            'RETURN':   self.insta_drop
        }

        key = list(key_actions)[action_number]
        key_actions[key]()

    def get_perf(self):
        return self.score, self.lines, self.pieces, sum(self.actions_list)

    def run(self, screen):

        if screen:
            pygame.init()
            pygame.key.set_repeat() # held keys are not repeated
            self.width = cell_size*(cols+8)
            self.height = cell_size*rows
            self.rlim = cell_size*cols
            self.bground_grid = [[ 8 if x%2 == y%2 else 0 for x in range(cols)] for y in range(rows)]
            self.default_font = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("TETRIS")
            pygame.event.set_blocked(pygame.MOUSEMOTION) # block mouse movements

        self.gameover = False

        clock = pygame.time.Clock()

        while 1:
            if screen:
                self.screen.fill(colors[0])
                if self.gameover:
                    self.center_msg("""Game Over!\n\nYour score: %d""" % self.score)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.rlim+1, 0), (self.rlim+1, self.height-1))
                    self.disp_msg("Score: %d\n\n#Lines: %d" % (self.score, self.lines),
                        (self.rlim + cell_size, cell_size * 2))
                    self.draw_matrix(self.bground_grid, (0,0))
                    self.draw_matrix(self.board, (0,0))
                    self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
                pygame.display.update()

            action = random.randint(0,5)
            if not self.gameover:
                self.step(action)
                self.actions += 1
            else:
                break

            clock.tick(maxfps)

if __name__ == '__main__':
    App = TetrisApp()
    App.run(True)
