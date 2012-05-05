import numpy as np
from itertools import product

from elements import Car
from geometry import X, Y, O

FLOAT = 'float32'

############# place fields ##############

NL1 = 5
NS1 = 2
N2 = 3

def grid(size_x, size_y, elongation,
         c = 2, x0 = 0, y0 = 0):
    """
    elongation: long, square, flat
    """
    if elongation == 'long':
        x = (size_x + c*(NL1-1) )/NL1
        y = (size_y + c*(NS1-1) )/NS1
    elif elongation == 'flat':
        x = (size_x + c*(NS1-1) )/NS1
        y = (size_y + c*(NL1-1) )/NL1
    elif elongation == 'square':
        x = (size_x + c*(N2-1) )/N2
        y = x
    i, j = 0, 0
    fields = []
    while 1:
        if i < size_x:
            if i: i -= (c+1)/2
            while 1:
                if j < size_y:
                    if j: j -= (c+1)/2
                    fields.append(
                        (x0+i, x0+min(i+x, size_x),
                         y0+j, y0+min(j+y, size_y) ) )
                    j += y
                else:
                    j = 0
                    break
            i += x 
        else: break
    return fields

def p_field(length, width, c = 2):
    """
    for s_shape race
    """
    fields = []
    # 1
    x0, y0 = 0, 0
    x, y = length+width, width
    fields += grid(x, y, 'long',
                   c, x0, y0)
    # 2
    x0, y0 = width+length, 0
    x, y = width, width
    fields += grid(x, y, 'square',
                   c, x0, y0)
    # 3
    x0, y0 = width+length, width
    x, y = width, length
    fields += grid(x, y, 'flat',
                   c, x0, y0)
    # 4
    x0, y0 = length+width, length+width
    x, y = width, width
    fields += grid(x, y, 'square',
                   c, x0, y0)
    # 5
    x0, y0 = width, width+length
    x, y = length, width
    fields += grid(x, y, 'long',
                   c, x0, y0)
    # 6
    x0, y0 = 0, length+width
    x, y = width, width
    fields += grid(x, y, 'square',
                   c, x0, y0)
    # 7
    x0, y0 = 0, length+2*width
    x, y = width, length
    fields += grid(x, y, 'flat',
                   c, x0, y0)
    # 8
    x0, y0 = 0, 2*length+2*width
    x, y = width, width
    fields += grid(x, y, 'square',
                   c, x0, y0)
    # 9
    x0, y0 = width, 2*length+2*width
    x, y = length+width, width
    fields += grid(x, y, 'long',
                   c, x0, y0)
    assert len(fields) == 5*10+4*9, 'wrong amount of cells'
    return fields

# 29    44    43    42    41
#
# 30   9 28 27 26 25 24   40
#     10             23
#     11   1 8 7     22
# 31       2 0 6          39
#     12   3 4 5     21
#     13             20
# 32  14 15 16 17 18 19   38
#
# 33    34    35    36    37

def v_field():
    fields = [(0,1, 0,1)]
    # 1st circle
    fields += [(-1+i,i, -1,0) for i in range(3)]
    fields += [(1,2, i,i+1) for i in range(2)]
    fields += [(-i,1-i, 1,2) for i in range(2)]
    fields += [(-1,0, 0,1)]
    # 2nd circle
    fields += [(-3+i,-1+i, -3,-1) for i in range(6)]
    fields += [(2,4, -2+i,i) for i in range(5)]
    fields += [(1-i,3-i, 2,4) for i in range(5)]
    fields += [(-3,-1, 1-i,3-i) for i in range(4)]
    # 3rd circle
    fields += [(-5+2*i,-2+2*i, -5,-2) for i in range(5)]
    fields += [(3,6, -3+2*i,2*i) for i in range(4)]
    fields += [(1-2*i,4-2*i, 3,6) for i in range(4)]
    fields += [(-5,-2, 1-2*i,4-2*i) for i in range(3)]
    return fields

class PlaceField:

    def __init__(self, length, width, c):
        self.p = p_field(length, width, c)
        self.v = v_field()

    def in_place_field(self, vector, field):
        return field[0] <= vector[0] < field[1] and\
            field[2] <= vector[1] < field[3]

    def where(self, p, v):
        p_idx = []
        for i in range(len(self.p) ):
            if self.in_place_field(p, self.p[i]):
                p_idx.append(i)
        v_idx = []
        for i in range(len(self.v) ):
            if self.in_place_field(v, self.v[i]):
                v_idx.append(i)
        return p_idx, v_idx

############# NN Driver #################

C = 2
ACTIONS = [O, X, Y, -X, -Y]

class NNDriver(Car):

    wo = 0.5
    wmax = 10
    sigma = 0.1
    tau = 5
    explo = 0.
    eta = 0.01
    speed = 0.1
    stop = -10
    wrong = -1
    r_win = 10
    nn_opening = '# NN agent\n'

    def __init__(self, race, name):
        Car.__init__(self, race)
        self.fields = PlaceField(
            race.circuit.length,
            race.circuit.width,
            C)
        self.decay = np.exp(-1./self.tau)
        self.grid = np.zeros(
            len(self.fields.p)*len(self.fields.v), bool)
        self.actions = np.zeros(len(ACTIONS), FLOAT)
        self.last_action = np.zeros(self.actions.shape, bool)
        self.load(name)
        self.e_trace = np.zeros(self.cx.shape, FLOAT)
        self.reward = 0

    def turn(self):
        self.get_reward()
        self.plasticity()
        self.where()
        self.propagate()
        if self.miss:
            self.miss -= 1
        else:
            if np.random.rand()<self.explo:
                idx = np.random.randint(len(ACTIONS) )
            else:
                idx = self.policy()
            self.v += ACTIONS[idx]
            self.last_action *= False
            self.last_action[idx] = True
        return self.miss

    def policy(self):
        max_a = self.actions.max()
        where = self.actions==max_a
        if (where).sum()>1:
            idx = []
            for i in range(len(where)):
                if where[i]:
                    idx.append(i)
            return idx[
                np.random.randint(len(idx) )]
        else:
            return list(self.actions).index(max_a)

    def where(self):
        self.grid *= False
        p_idx, v_idx = self.fields.where(
            self.p, self.v)
        it = product(p_idx, v_idx)
        lv = len(self.fields.v)
        for p_i, v_i in it:
            self.grid[p_i*lv+v_i] = True
        if self.race.display:
            print 'cell nb: ', np.where(self.grid)[0]/len(self.fields.v)

    def propagate(self):
        self.actions = np.dot(self.cx, self.grid)
        if self.race.display:
            print '0, S, E, N, W: ', self.actions

    def plasticity(self):
        # reward prediction
        self.cx[self.last_action, self.grid] += self.eta*(
            self.reward-self.cx[self.last_action, self.grid])
        # eligibility trace
        if self.last_action.any() and self.grid.any():
            self.cx += self.eta*self.e_trace*self.reward
                # self.cx[self.last_action, self.grid].mean()
        self.cx.clip(-self.wmax, self.wmax, self.cx)
        self.e_trace *= self.decay
        self.e_trace[self.last_action, self.grid] += 1

    def get_reward(self):
        self.reward = 0
        cl = self.race.circuit.length
        cw = self.race.circuit.width
        # not moving
        v = abs(self.v).sum()
        if not v:
            self.reward += self.stop
        # speed
        else:
            self.reward += self.speed*v**2
        lv = len(self.fields.v)
        # down
        if (self.grid[:10*lv].any() or
            self.grid[-10*lv:].any() ) and\
            self.v[0]<0:
            self.reward *= self.wrong
        # up
        if self.grid[38*lv:48*lv].any() and\
                self.v[0]>0:
            self.reward *= self.wrong
        # right
        if (self.grid[19*lv:29*lv].any() or
            self.grid[57*lv:67*lv].any() ) and\
            self.v[1]<0:
            self.reward *= self.wrong
        if self.race.display:
            print 'reward: ', self.reward

    def end(self):
        Car.end(self)
        self.plasticity()
        if self.race.display:
            save = raw_input('save [y/N]? ')
            if save == 'y':
                self.save(raw_input('name: ') )

    def win(self, pos):
        self.reward = self.r_win
        Car.win(self, pos)

    def save(self, name):
        print 'saving NNAgent '+name
        f = open(name + '.nna', 'w')
        f.write(self.nn_opening)
        x, y = self.cx.shape
        for i in range(x):
            for j in range(y):
                f.write('%i %i %.6f\n'
                        %(i, j, self.cx[i,j]) )
        f.close()
        print 'done'

    def load(self, name):
        self.cx = np.zeros(
            self.actions.shape+self.grid.shape, FLOAT)
        try:
            name += '.nna'
            print 'opening NNAgent '+name
            f = open(name, 'r')
            assert f.readline() == self.nn_opening, 'wrong format'
            for line in f:
                line = line.split(' ')
                i, j = int(line[0]), int(line[1])
                w = float(line[2])
                self.cx[i, j] = w
            f.close()
            print 'closed'
        except IOError:
            print 'initiating naively...'

############# random agent ################

class RandDriver(Car):

    def __init__(self, race, *args):
        Car.__init__(self, race)

    def turn(self):
        """
        one turn
        """
        self.v += self.rand_dir()
