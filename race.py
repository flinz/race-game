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

X = np.array([1, 0])
Y = np.array([0, 1])
O = 0*X

COLORS = [
    (0.0, 0.0, 1.0),   # 'b'
    (0.0, 0.5, 0.0),   # 'g'
    (1.0, 0.0, 0.0),   # 'r'
    (0.0, 0.75, 0.75), # 'c'
    (0.75, 0, 0.75),   # 'm'
    (0.75, 0.75, 0),   # 'y'
    (0.0, 0.0, 0.0),   # 'k'
    (1.0, 1.0, 1.0),   # 'w'
    ]

############### general algorithms ######

def find_path(pos, aim):
    """
    the path from pos to aim
    returns a list of positions
    """
    vel = aim - pos
    dx, dy = tuple(vel)
    way = np.clip(vel, -1, 1)
    vel = abs(vel)
    drc = (X, Y)
    idx = 0 + 1*(vel[1]>vel[0])
    path = []
    if vel[1-idx] == 0:
        while pos[idx] < aim[idx]:
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
    return path

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

    def register(self, race):
        self.race = race

    def init_history(self):
        history = []
        for _ in xrange(self.race.t):
            history.append(None)
        history.append(self.p)
        self.history = history

    def rec_position(self):
        """
        records actual position
        """
        self.history.append(tuple(self.p) )

    def fill_history(self):
        for _ in xrange(
            self.race.t-len(self.history) ):
            history.append(None)

############### players #################

class Player(Element):

    N_MISSILE = 2
    PLAY_TIME = 10
    P_JITTER = 0.05

    def __init__(self, race, index, color,
                 position):
        Element.__init__(self, race, index, color,
                         position, O.copy() )                         
        self.play = 0
        self.missile = self.N_MISSILE

    def reset_pos(self):
        """
        resets position
        """
        self.p = np.array(self.history[0])

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
        dirs = [X, -X, Y, -Y, 0*X]
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
            return -X, False
        elif tmp in LEFT:
            return -Y, False
        elif tmp in RIGHT:
            return Y, False
        elif tmp in DOWN:
            return X, False
        elif tmp in STOP:
            return O, False
        elif tmp in MISSILE:
            return O, True
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
            direction = O
        else:
            direction = self.rand_dir()
        return direction

    def rand_dir(self):
        """
        returns random direction
        """
        sign = (2*np.random.randint(0,2)-1)
        direction = X + (Y-X)*(
            np.random.rand()<0.5)
        return sign*direction

    def hit_speed(self):
        v = abs(self.v).sum()
        self.play = np.ceil(
            -(1+v)+np.sqrt(1+2*v*(v+1)))
        self.v = O
    
    def hit(self, pos):
        """
        hits wall or other element
        """
        print 'you crashed...'
        walls = True
        for player in self.race.players:
            if tuple(player.p) == pos:
                walls = False
                # to self
                self.hit_speed()
                # graph
                self.race.graphics.draw(
                    explosion = (pos, self.play) )
                # to player
                self.race.move(player)
        for missile in self.race.missiles:
            if tuple(missile.p) == pos:
                walls = False
                # to missile
                self.race.missiles.remove(
                    missile)
                self.race.lost_missiles.append(missile)
                # graph
                self.race.graphics.draw(
                    explosion = (pos, missile.SIZE) )
                # to self
                self.p = np.array(pos)
                self.v = self.rand_dir()
        if walls:
            # to self
            self.hit_speed()
            # graph
            self.race.graphics.draw(
                explosion = (pos, self.play) )

############## missile ##################

class Missile(Element):

    V = 10
    SIZE = 3

    def __init__(self, race, index, car_position,
                 car_velocity):
        vx, vy = tuple(car_velocity)
        v = np.sqrt( (car_velocity**2).sum() )
        velocity = np.array([round(vx*self.V/v), round(vy*self.V/v)])
        Element.__init__(self, race, index, COLORS[6],
                         car_position, velocity)

    def hit(self, pos):
        for player in self.race.players:
            if pos == tuple(player.p):
                # to player
                player.v = player.rand_dir()
        for missile in self.race.missiles:
            if missile != self and\
                    tuple(missile.p) == pos:
                # to missile
                self.race.missiles.remove(
                    missile)
                self.race.lost_missiles.append(missile)
        # to self
        self.race.missiles.remove(self)
        self.race.lost_missiles.append(missile)
        # graph
        self.race.graphics.draw(
            explosion = (pos, missile.SIZE) )
    
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

    SGM = 2.
    START_ZONE = -5.

    def __init__(self, shape, length, width):
        self.circuit, self.start, self.end = \
            CIRCUITS[shape](length, width)
        self.obstacles = np.zeros(self.circuit.shape, bool)

    def add_obstacles(self, density):
        self.obstacles = (
            (np.random.rand(*self.circuit.shape)<prob))

    def filter_obstacles(self, threshold, show = False):
        obstacles = self.obstacles.copy().astype(int)
        gauss = 1/np.sqrt(2*np.pi)/self.SGM
        func = lambda x: np.exp(-0.5*x**2/self.SGM**2)*gauss
        # change zone !!!
        for pos in self.start:
            obstacles[pos] = self.START_ZONE
        X, Y = self.circuit.shape
        mat = np.zeros( (X, Y), float)
        it = product(xrange(2*X), xrange(2*Y))
        for x,y in it:
            coef = func(np.sqrt( (x-X)**2 + (y-Y)**2) )
            if coef>1e-6:
                mat[max(0,x-X):min(x,X), max(0,y-Y):min(y,Y)] += coef *\
                    obstacles[max(0,X-x): min(2*X-x,X),
                              max(0,Y-y): min(2*Y-y,Y)]
        rand = np.random.rand(*mat.shape)
        self.obstacles = rand < threshold*mat
        if show:
            plt.figure()
            plt.subplot(221)
            plt.imshow(obstacles)
            plt.subplot(222)
            plt.imshow(final)
            plt.subplot(212)
            plt.imshow(mat)
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

class Graphics:

    red = np.array([0.7, 0, 0])
    yellow = (0.9, 0.8, 0)
    unif = np.ones(3)
    white = 0.8*unif
    grey = 0.4*unif

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
            cir.circuit - cir.obstacles] = self.white
        
    def show(self):
        self.graph.set_data(self.image)
        plt.draw()
        sleep(0.3)

    def frame(self, color):
        self.image[ [0,-1] ] = color
        self.image[:, [0,-1] ] = color

    def explosion(self, pos, size):
        x, y = replace_tuple(pos)
        it = product(range(x-size,x+size+1),
                     range(y-size,y+size+1))
        size_x, size_y = self.image.shape
        for a,b in it:
            if np.sqrt( (a-x)**2+(b-y)**2) <= size and\
                    (0 <= a < size_x) and \
                    (0 <= b < size_y):
                self.image[a, b] = self.yellow

    def place(self, element):
        self.image[tuple(element.p+X+Y)] = element.color

    def history(self, t):
        for element in self.race.players + self.race.lost_missiles:
            for pos in element.history[:t]:
                if pos:
                    pos = replace_tuple(pos)
                    self.image[pos] = element.color

    def cross(self, positions):
        for pos in positions:
            pos = replace_tuple(pos)
            self.image[pos] *= 0.3
            self.image[pos] += \
                np.array(self.red)

    def draw(self, frame = None, cross = None, explosion = None, history = None):
        self.reset_image()
        for element in self.race.players + self.race.missiles:
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

class Race:

    def __init__(self, n, shape, length, width):
        """
        n: # players
        """
        self.t = 0
        self.n = n
        self.players = []
        self.missiles = []
        self.circuit = Circuit(shape, length, width)
        starting_blocks = copy(self.circuit.start)
        np.random.shuffle(starting_blocks)
        for idx in xrange(n):
            color = COLORS[idx]
            # color = np.random.rand(3)
            # while color.std()<0.1:
            #     color = np.random.rand(3)
            self.players.append(
                Player(self, idx, color,
                       np.array(starting_blocks[idx]) ) )
        self.graphics = Graphics(self, self.circuit)

    def reset_pos(self):
        for player in players:
            player.reset_pos()
    
    def rec_pos(self):
        """
        records all positions
        """
        for player in self.players:
            player.rec_position()
        for missile in self.missiles:
            missile.rec_position()

    def add_missile(self, player):
        idx = len(self.missile)
        self.missiles.append(
            Missile(self, idx, player.p,
                    player.v) )
        self.move(self.missiles[-1] )
            
    def available(self, element, pos):
        ok = self.circuit.available(pos)
        for player in self.players:
            if player != element and\
                    tuple(player.p) == pos:
                ok = False
        for missile in self.missiles:
            if missile != element and\
                    tuple(missile.p) == pos:
                ok = False
        return ok

    def move(self, element):
        """
        one move for 'element'
        """
        elmnt_p = element.p
        aim = elmnt_p + element.v
        path = find_path(elmnt_p, aim)
        for pos in path[1:]:
            prev_pos = tuple(elmnt_p)
            # skip a corner
            # problem !!!
            print pos, prev_pos
            if manhattan_dist(pos, prev_pos) > 1:
                ways = insert(pos, prev_pos)
                if not (
                    self.available(element, ways[0]) or\
                        self.available(element, ways[1]) ):
                    np.random.shuffle(ways)
                    element.hit(ways[0])
                    break
                else:
                    for way in ways:
                        if self.circuit.win_check(way):
                            running = False
                            elmnt_p = way
                            break
            # hit
            if not self.available(element, pos):
                element.hit(pos)
                break
            # win ?
            if self.circuit.win_check(pos):
                running = False
                elmnt_p = np.array(pos)
                break
            # advance
            elmnt_p = np.array(pos)

    def turn(self):
        win = False
        for player in self.players:
            if not player.play:
                print '- - - - - - - - - - -\nplayer %i turn' \
                    %player.id
                player.turn()
            else:
                print '- - - - - - - - - - -'
                print 'player %i misses %i turn' \
                    %(player.id, player.play)
                player.play -= 1
            self.move(player)
        for missile in self.missiles:
            self.move(missile)
        self.rec_pos()
        for player in self.players:
            if self.circuit.win_check(tuple(player.p) ):
                win = True
        self.graphics.draw()
        self.t += 1
        return win
            
    def run(self):
        running = True
        while running:
            running = not self.turn()
        self.reset_pos()
        for missile in self.lost_missiles:
            missile.fill_history()
        while 1:
            tmp = raw_input('for no replay, press "q"')
            if tmp == 'q':
                sys.exit()
            self.replay()

    def replay(self):
        for i in xrange(len(self.players[0].history)+1):
            self.graphics.draw(history = i)

############## main ##############

def usage():
    print '-n # players'
    print '-r race name (straight, s_shape)'
    print '-l race length'
    print '-w race width'
    print '-o obstacles density'
    print '-f filter threshold'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hn:r:l:w:o:f:',
                                   ['help'])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print "option -a not recognized"
        usage()
        sys.exit(2)
    n = 2
    length = 50
    width = 25
    density = 0
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
        elif o == '-f':
            if not density: density = 0.01
            threshold = float(a)
        else:
            assert False, 'unhandled option'
    race = Race(n, shape, length, width)
    if density:
        race.circuit.add_obstacles(density)
    if threshold:
        race.circuit.filter_obstacles(threshold, True)
    race.run()

if __name__ == '__main__':
    main()
