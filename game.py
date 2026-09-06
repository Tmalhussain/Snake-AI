import numpy as np
import random

moves = {'L': np.array([-1.,0.]),
    'R': np.array([1.,0.]),
    'U': np.array([0.,-1.]),
    'D': np.array([0.,1.])}

actions = {0: 'L', 1: 'R', 2: 'U', 3: 'D'}

initial_playersize = 4

class apple(object):
    def __init__(self, gridsize, occupied=()):
        self.gridsize = gridsize
        self.score = 0
        self.pos = np.zeros(2)
        self.respawn(occupied)
    def respawn(self, occupied=()):
        free = [(x, y) for x in range(self.gridsize) for y in range(self.gridsize) if (x, y) not in occupied]
        if not free:
            return False
        self.pos = np.array(random.choice(free), dtype=float)
        return True
    def eat(self, occupied=()):
        self.score += 1
        return self.respawn(occupied)

class snake(object):
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.pos = np.array([gridsize//2, gridsize//2]).astype('float')
        self.prevpos = [self.pos.copy()]
        self.dir = np.array([1.,0.])
        self.len = initial_playersize
    def move(self):
        self.pos += self.dir
        self.prevpos.append(self.pos.copy())
        self.prevpos = self.prevpos[-self.len:]
    def cells(self):
        return {tuple(i) for i in self.prevpos}
    def dead(self):
        if not (0 <= self.pos[0] < self.gridsize):
            return True
        if not (0 <= self.pos[1] < self.gridsize):
            return True
        cells = [tuple(i) for i in self.prevpos]
        if len(set(cells)) != len(cells):
            return True
        return False
    def won(self):
        return self.len >= self.gridsize**2

class GameEnv(object):
    APPLE_REWARD = 1.
    DEATH_REWARD = -1.
    STEP_REWARD = -0.01

    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.resetgame()

    def resetgame(self):
        self.snake = snake(self.gridsize)
        self.apple = apple(self.gridsize, occupied=self.snake.cells())
        self.gameover = False
        self.win = False
        self.truncated = False
        self.timeSinceApple = 0
        return self.get_state()

    def steplimit(self):
        return 100 + 50 * (self.snake.len - initial_playersize)

    def get_boardstate(self):
        return [self.snake.pos.copy(),
                self.snake.dir.copy(),
                [i.copy() for i in self.snake.prevpos],
                self.apple.pos.copy(),
                self.apple.score,
                self.gameover]

    def info(self):
        return {'score': self.apple.score,
                'length': self.snake.len,
                'won': self.win,
                'truncated': self.truncated}

    def update(self, move):
        if self.gameover:
            return self.get_state(), 0., True, self.info()

        d = moves[actions[move]]
        if not np.array_equal(d, -self.snake.dir):
            self.snake.dir = d

        self.snake.move()
        self.timeSinceApple += 1
        reward = self.STEP_REWARD

        if self.snake.dead():
            self.gameover = True
            reward = self.DEATH_REWARD
        elif np.array_equal(self.snake.pos, self.apple.pos):
            self.snake.len += 1
            self.timeSinceApple = 0
            reward = self.APPLE_REWARD
            if not self.apple.eat(self.snake.cells()) or self.snake.won():
                self.gameover = True
                self.win = True
        elif self.timeSinceApple >= self.steplimit():
            self.gameover = True
            self.truncated = True

        return self.get_state(), reward, self.gameover, self.info()

    def state_shape(self):
        return (2, self.gridsize + 2, self.gridsize + 2)

    def get_state(self):
        size = self.gridsize + 2
        state = np.zeros((2,size,size),dtype = np.float32)
        state[1,0,:] = 1.
        state[1,-1,:] = 1.
        state[1,:,0] = 1.
        state[1,:,-1] = 1.
        n = len(self.snake.prevpos)
        growing = n < self.snake.len
        for i,p in enumerate(self.snake.prevpos):
            state[0,int(p[0])+1,int(p[1])+1] = (i+1)/n
            if i > 0 or growing:
                state[1,int(p[0])+1,int(p[1])+1] = 1.
        state[0,int(self.apple.pos[0])+1,int(self.apple.pos[1])+1] = -1.
        return state
