import numpy as np
import matplotlib.pyplot as plt
from itertools import product

############## circuit ##################

def straight(length, width):
    race = np.ones((length,width), bool)
    start = [(0,i) for i in range(width)]
    end = [(length-1,i) for i in range(width)]
    return (race, start, end)

def s_shape(length, width):
    race = np.zeros((length+2*width,5*width), bool)
    race[:,0:width] = True
    race[:,2*width:3*width] = True
    race[:,4*width:5*width] = True
    race[-width:,width:2*width] = True
    race[:width,3*width:4*width] = True
    start = [(0,i) for i in range(0,width)]
    end = [(length+2*width-1,i) for i in range(4*width,5*width)]
    return (race, start, end)

CIRCUITS = {
    'straight': straight,
    's_shape': s_shape
    }

class Circuit:

    start_zone = -5.
    opening = '# circuit format copyright Ziegler 2012\n'

    def __init__(self, shape, length, width, name):
        self.shape = shape
        self.length = length
        self.width = width
        if name:
            self.read(name)
        else:
            self.create_circuit(shape, length, width)

    def create_circuit(self, shape, length, width):
        self.circuit, self.start, self.end = \
            CIRCUITS[shape](length, width)
        self.obstacles = np.zeros(self.circuit.shape, bool)

    def add_obstacles(self, density):
        self.obstacles = (
            (np.random.rand(*self.circuit.shape)<density) )

    def filter_obstacles(self, threshold, sgm = 2., show = False):
        obstacles = self.obstacles.copy().astype(float)
        gauss = 1/np.sqrt(2*np.pi)/sgm
        func = lambda x: np.exp(-0.5*x**2/sgm**2)*gauss
        # change zone !!!
        for pos in self.start:
            obstacles[pos] = self.start_zone
        sx, sy = self.circuit.shape
        mat = np.zeros( (sx, sy), float)
        it = product(xrange(2*sx), xrange(2*sy))
        for x,y in it:
            coef = func(np.sqrt( (x-sx)**2 + (y-sy)**2) )
            if coef>1e-6:
                mat[max(0,x-sx):min(x,sx),
                    max(0,y-sy):min(y,sy)] += coef *\
                    obstacles[max(0,sx-x): min(2*sx-x,sx),
                              max(0,sy-y): min(2*sy-y,sy)]
        rand = np.random.rand(*mat.shape)
        self.obstacles = rand < threshold*mat
        if show:
            plt.figure()
            plt.subplot(221)
            plt.imshow(obstacles, **plot_dic)
            plt.subplot(222)
            plt.imshow(self.obstacles, **plot_dic)
            plt.subplot(212)
            plt.imshow(mat, **plot_dic)
            plt.colorbar()

    def in_frame(self, pos):
        size_x, size_y = self.circuit.shape
        return (
            (0 <= pos[0] < size_x) and
            (0 <= pos[1] < size_y) )
            
    def available(self, pos):
        if not self.in_frame(pos):
            return False
        else:
            return self.circuit[pos]*(
                not self.obstacles[pos] )

    def win_check(self, pos):
        return (pos in self.end)

    def write(self, name):
        print 'writing...'
        f = open(name + '.cir', 'w')
        f.writelines([self.opening,
                      self.shape + ' %i %i\n'%(self.length, self.width)])
        sx, sy = self.obstacles.shape
        it = product(xrange(sx), xrange(sy))
        for x,y in it:
            if self.obstacles[x, y]:
                f.write('%i %i\n'%(x, y))
        f.close()
        print 'done'

    def read(self, name):
        name += '.cir'
        print 'opening', name, '...'
        f = open(name, 'r')
        assert f.readline() == self.opening, 'wrong format'
        line = f.readline()
        line = line.split(' ')
        self.shape = line[0]
        self.length = int(line[1])
        self.width = int(line[2])
        self.create_circuit(self.shape, self.length, self.width)
        for line in f:
            line = line.split(' ')
            self.obstacles[line[0], line[1]] = True
        f.close()
        print 'closed'
