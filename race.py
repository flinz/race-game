# remove np.arrays -> design tools for adding tuples
# new classes: image
# move functions where they belong

import numpy as np
import getopt, sys
from time import sleep, time
import matplotlib.pyplot as plt
from itertools import product
from copy import copy

plt.ion()
plot_dic={'interpolation':'nearest'}

UP = '8w'
LEFT = '4a'
RIGHT = '6d'
DOWN = '2x'
STOP = '5s'
QUIT = 'quit'
MISSILE = 'm'

X = np.array([1, 0], int)
Y = np.array([0, 1], int)
O = np.zeros(2, int)

############### general algorithms ######

def find_path(position, velocity):
    """
    the path from pos to aim
    returns a list of positions
    """
    pos = position.copy()
    vel = velocity.copy()
    aim = pos + vel
    dx, dy = tuple(vel)
    way = np.clip(vel, -1, 1)
    vel = abs(vel)
    drc = (X, Y)
    idx = 0 + 1*(vel[1]>vel[0])
    path = []
    if vel[1-idx] == 0:
        while pos[idx] != aim[idx]:
            path.append(tuple(pos) )
            pos += way[idx]*drc[idx]
    else:
        e = vel[idx]
        vel *= 2
        while 1:
            path.append(tuple(pos) )
            pos += way[idx]*drc[idx]
            if pos[idx] == aim[idx]: break
            e -= vel[1-idx]
            if e < 0:
                pos += way[1-idx]*drc[1-idx]
                e += vel[idx]
    path.append(tuple(aim) )
    return path[1:]

def manhattan_dist(p1, p2):
    return abs(p2[0]-p1[0]) + abs(p2[1]-p1[1])

def insert(p1, p2):
    assert manhattan_dist(p1, p2) == 2, \
        'positions too far away'
    return [(p1[0], p2[1]), (p2[0], p1[1])]

def replace_tuple(tup):
    return tup[0]+1, tup[1] +1

############### base class ##############

class Element:

    def __init__(self, race, index, color,
                 position, velocity):
        self.id = index
        self.color = color
        self.p = position
        self.v = velocity
        self.register(race)
        self.init_history()
        self.avoid = False # !!!
        self.play = 0

    def register(self, race):
        self.race = race

    def init_history(self):
        history = []
        for _ in xrange(self.race.t):
            history.append( () )
        history.append(tuple(self.p) )
        self.history = history

    def fill_history(self):
        for _ in xrange(
            self.race.t-len(self.history) ):
            self.history.append( () )

    def rec_position(self):
        """
        records actual position
        """
        self.history.append(tuple(self.p) )

    def turn(self):
        pass

    def win(self, pos):
        self.p = np.array(pos, int)
        self.end()

    def end(self):
        self.rec_position()
        self.race.out(self)
        
############### players #################

class Player(Element):

    N_MISSILE = 10
    PLAY_TIME = 10
    P_JITTER = 0.05
    V_IN = 0.5
    V_INTO = 0.5

    def __init__(self, race, index, color,
                 position):
        Element.__init__(self, race, index, color,
                         position, O.copy() )   
        self.missile = self.N_MISSILE

    def turn(self):
        """
        one turn
        """
        t = time()
        dv, missile = self.next_move()
        if (time()-t) > self.PLAY_TIME:
            dv = self.jitter()
        if missile and self.missile > 0:
            self.race.add_missile(self)
            self.missile -= 1
        self.v += dv

    def next_pos(self):
        """
        returns a list of possible
        moves
        """
        dirs = [X, -X, Y, -Y, O]
        positions = []
        for d in dirs:
            pos = self.p + self.v + d
            if self.race.circuit.in_frame(pos):
                positions.append(tuple(pos) )
        return positions

    def next_move(self):
        """
        asks what to do next
        """
        self.race.graphics.draw(
            frame = self.color,
            cross = self.next_pos() )
        tmp = raw_input(
            '\nnext move? (use numpad or w-a-s-d-x) ')
        if tmp in UP:
            return -X.copy(), False
        elif tmp in LEFT:
            return -Y.copy(), False
        elif tmp in RIGHT:
            return Y.copy(), False
        elif tmp in DOWN:
            return X.copy(), False
        elif tmp in STOP:
            return O.copy(), False
        elif tmp in MISSILE:
            return O.copy(), True
        elif tmp == QUIT:
            sys.exit()
        else:
            return self.next_move()

    def jitter(self):
        """
        returns random direction with
        probability P_JITTER (else 0)
        """
        if np.random.rand() < 1-self.P_JITTER:
            direction = O.copy()
        else:
            direction = self.rand_dir()
        return direction

    def rand_dir(self):
        """
        returns random direction
        """
        sign = (2*np.random.randint(0,2)-1)
        direction = X.copy() + (Y.copy()-X.copy() )*(
            np.random.rand()<0.5)
        return sign*direction

    def hit_speed(self):
        v = abs(self.v).sum()
        self.play = int(np.ceil(
                -(1+v)+np.sqrt(1+2*v*(v+1) ) ) )
        v_hit = self.v.copy()
        self.v = O.copy()
        return v_hit

    def hit(self, what):
        """
        hits wall or other element
        """
        print 'you crashed...'
        if what.__class__ == Player or what == 'wall':
            v = abs(self.v).sum()
            self.play = int(np.ceil(
                    -(1+v)+np.sqrt(1+2*v*(v+1) ) ) )
            self.v = O.copy()
        elif what.__class__ == Missile:
            self.be_hit(what)

    def be_hit(self, element, v):
        if element.__class__ == Player:
            self.v = (self.V_IN*v +
                      self.V_INTO*self.v).round()
            self.race.move(self)
        if element.__class__ == Missile:
            pass
        # if hasn't crashed
        if not self.play:
            self.v = self.rand_dir()

    def win(self, pos):
        Element.win(self, pos)
        print '\nyeah\n'
        self.race.arrived_players += 1

############## missile ##################

class Missile(Element):

    V = 10
    SIZE = 3

    def __init__(self, race, index, car_position,
                 car_velocity):
        vx, vy = tuple(car_velocity)
        v = np.sqrt( (car_velocity**2).sum() )
        if v == 0:
            vx = np.random.rand()
            vy = np.random.rand()
            v = np.sqrt(vx**2+vy**2)
        velocity = np.array([round(vx*self.V/v),
                             round(vy*self.V/v)], int)
        Element.__init__(self, race, index, 'black',
                         car_position.copy(), velocity)

    def hit(self, what):
        self.end()

    def be_hit(self, element, v):
        self.end()
        
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

    START_ZONE = -5.

    def __init__(self, shape, length, width):
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
            obstacles[pos] = self.START_ZONE
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

############## graphics #################

COLORS = {
    'blue': (0.0, 0.0, 1.0),
    'green': (0.0, 0.5, 0.0),
    'red': (1.0, 0.0, 0.0),
    'cyan': (0.0, 0.75, 0.75),
    'magenta': (0.75, 0, 0.75),
    'yellow': (0.75, 0.75, 0),
    'black': (0.0, 0.0, 0.0),
    'white': (1.0, 1.0, 1.0),
    }

def scale(color, k):
    assert len(color) == 3, 'not a RGB color'
    return k*color[0], k*color[1], k*color[2]

class Graphics:

    red = scale(COLORS['red'], 0.7)
    yellow = (0.9, 0.8, 0)
    white = scale(COLORS['white'], 0.8)
    grey = scale(COLORS['white'], 0.4)

    def __init__(self, race, circuit):
        self.race = race
        self.circuit = circuit
        x, y = circuit.circuit.shape
        self.image = np.zeros( (x+2, y+2, 3) )
        self.reset_image()
        plt.figure()
        self.graph = plt.imshow(self.image, **plot_dic)

    def reset_image(self):
        self.image[:, :] = self.grey
        cir = self.circuit
        self.image[1:-1, 1:-1][
            cir.circuit*(-cir.obstacles)] = self.white
        
    def show(self):
        self.graph.set_data(self.image)
        plt.draw()
        sleep(0.3)

    def frame(self, color):
        self.image[ [0,-1] ] = COLORS[color]
        self.image[:, [0,-1] ] = COLORS[color]

    def explosion(self, pos, size):
        x, y = replace_tuple(pos)
        it = product(range(x-size,x+size+1),
                     range(y-size,y+size+1))
        size_x, size_y = self.image.shape[:2]
        for a,b in it:
            if np.sqrt( (a-x)**2+(b-y)**2) <= size and\
                    (0 <= a < size_x) and \
                    (0 <= b < size_y):
                self.image[a, b] = self.yellow

    def place(self, element):
        self.image[tuple(element.p+X+Y)] = COLORS[element.color]

    def history(self, t):
        for element in self.race.done:
            for pos in element.history[:t]:
                if len(pos) > 0:
                    pos = replace_tuple(pos)
                    self.image[pos] = COLORS[element.color]

    def cross(self, positions):
        for pos in positions:
            pos = replace_tuple(pos)
            self.image[pos] *= 0.3
            self.image[pos] += \
                np.array(self.red)

    def draw(self, frame = None, cross = None, explosion = None,
             history = None):
        self.reset_image()
        for element in self.race.playing:
            self.place(element)
        if frame:
            self.frame(frame)
        if cross:
            self.cross(cross)
        if explosion:
            self.explosion(*explosion)
        if history:
            self.history(history)
        self.show()
            
        
############## race #####################

COLOR_ORDER = [
    'blue',
    'green',
    'red',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'white',
    ]

class Race:

    def __init__(self, n, shape, length, width):
        """
        n: # players
        """
        self.t = 0
        self.n = n
        self.playing = []
        self.done = []
        self.arrived_players = 0
        self.circuit = Circuit(shape, length, width)
        starting_blocks = copy(self.circuit.start)
        np.random.shuffle(starting_blocks)
        for idx in xrange(n):
            color = COLOR_ORDER[idx]
            position = np.array(starting_blocks[idx], int)
            # color = np.random.rand(3)
            # while color.std()<0.1:
            #     color = np.random.rand(3)
            self.playing.append(
                Player(self, idx, color, position) )
        self.graphics = Graphics(self, self.circuit)

    def out(self, element):
        try:
            self.playing.remove(element)
        except ValueError, err:
            pass
        if element not in self.done:
            self.done.append(element)

    def close(self):
        for element in self.playing:
            self.out(element)
        self.fill_history()

    def fill_history(self):
        for element in self.done:
            element.fill_history()
            
    def rec_pos(self):
        """
        records all positions
        """
        for element in self.playing:
            element.rec_position()

    def add_missile(self, player):
        idx = len(self.playing)
        missile = Missile(self, idx, player.p,
                          player.v)
        self.playing.insert(
            self.playing.index(player),
            missile)
        self.move(missile)
            
    def available(self, element, pos):
        on_the_way = None
        if not self.circuit.available(pos):
            on_the_way = 'wall'
        for check in self.playing:
            if check != element and\
                    tuple(check.p) == pos:
                on_the_way = check
        return on_the_way

    def move(self, element):
        """
        one move for 'element'
        """
        path = find_path(element.p, element.v)
        for pos in path:
            prev_pos = tuple(element.p)
            # skip a corner
            stop = False
            if manhattan_dist(pos, prev_pos) > 1:
                ways = insert(pos, prev_pos)
                ways = [(way, self.available(element, way) )
                        for way in ways]
                # both blocked
                if ways[0][1] and ways[1][1]:
                    np.random.shuffle(ways)
                    self.hit(element, *ways[0])
                    stop = True
                    break
                else:
                    for way in ways:
                        if self.circuit.win_check(way):
                            element.win(way)
                            stop = True
                            break
            if stop: break
            # hit
            on_the_way = self.available(element, pos)
            if on_the_way:
                self.hit(element, pos, on_the_way)
                break
            # win ?
            if self.circuit.win_check(pos):
                element.win(pos)
                break
            # advance
            element.p = np.array(pos, int)

    def hit(self, element, pos, what):
        """
        hits wall or other element
        """
        # to element
        v = element.v.copy()
        element.hit(what)
        # graph
        if what.__class__ == Missile or\
                element.__class__ == Missile:
            size = Missile.SIZE
        else:
            size = element.play
        self.graphics.draw(
            explosion = (pos, size) )
        # to what
        if what != 'wall':
            what.be_hit(element, v)

    def turn(self):
        for element in self.playing:
            if not element.play:
                print '- - - - - - - - - - -'
                print 'player (missile) %i turn' \
                    %element.id
                element.turn()
            else:
                print '- - - - - - - - - - -'
                print 'player %i misses %i turn' \
                    %(element.id, element.play)
                element.play -= 1
            self.move(element)
        self.rec_pos()
        self.graphics.draw()
        self.t += 1
            
    def run(self):
        self.graphics.draw()
        tmp = raw_input('press RETURN to start, Q to quit ')
        if tmp == 'q':
            sys.exit()
        while self.arrived_players < min(3, max(self.n-1, 1)):
            self.turn()
        self.close()
        while 1:
            tmp = raw_input('press Q to quit ')
            if tmp == 'q':
                sys.exit()
            self.replay()

    def replay(self):
        for i in xrange(len(self.done[0].history)+1):
            self.graphics.draw(history = i)

############## main ##############

def usage():
    print '\n-n # players'
    print '-r race name (straight, s_shape)'
    print '-l race length'
    print '-w race width'
    print '-o obstacles density'
    print '-s filter sigma'
    print '-f filter threshold'
    print '\n--help   prints this'
    print '--filter automatic parameter set'
    print '         density = 0.01, sigma = 1., threshold = 5.\n'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hn:r:l:w:o:f:s:',
                                   ['help', 'filter'])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print "option -a not recognized"
        usage()
        sys.exit(2)
    n = 2
    length = 50
    width = 25
    density = 0
    sigma = 0
    threshold = 0
    shape = 'straight'
    for o, a in opts:
        if o == '-n':
            n = int(a)
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        elif o == '-l':
            length = int(a)
        elif o == '-w':
            width = int(a)
        elif o == '-o':
            density = float(a)
        elif o == '-r':
            shape = a
        elif o == '-s':
            if not density: density = 0.01
            if not threshold: threshold = 5.
            sigma = float(a)
        elif o == '-f':
            if not density: density = 0.01
            if not sigma: sigma = 1.
            threshold = float(a)
        elif o == '--filter':
            density = 0.01
            sigma = 1.
            threshold = 5.
        else:
            assert False, 'unhandled option'
    race = Race(n, shape, length, width)
    if density:
        race.circuit.add_obstacles(density)
    if threshold or sigma:
        race.circuit.filter_obstacles(threshold, sigma) # , True)
    race.run()

if __name__ == '__main__':
    main()
