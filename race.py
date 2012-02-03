# remove np.arrays -> design tools for adding tuples
# new classes: player, image

import numpy as np
import getopt, sys
from time import sleep, time
import matplotlib.pyplot as plt
from itertools import product

plt.ion()
plot_dic={'interpolation':'nearest'}

UP = '8w'
LEFT = '4a'
RIGHT = '6d'
DOWN = '2x'
STOP = '5s'
QUIT = 'quit'

X = np.array([1, 0])
Y = np.array([0, 1])
O = 0*X

P_JITTER = 0.05

PLAY_TIME = 10

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

############### players #################

class Player:

    def __init__(self, index, color, position):
        self.id = index
        self.color = color
        self.p = position
        self.v = O.copy()
        self.history = [tuple(position)]
        self.play = 0

    def rec_position(self):
        """
        records actual position
        """
        self.history.append(tuple(self.p) )

    def reset_pos(self):
        self.p = np.array(self.history[0])

############## race #####################

class Race:

    def __init__(self, n, race, start, end, color_style = 'rgb'):
        """
        N: # players
        race: race - boolean matrix
        start, end : [(xi, yi)]
        """
        self.n = n
        self.players = []
        np.random.shuffle(start)
        for idx in xrange(n):
            if color_style == 'rgb':
                color = COLORS[idx]
            else:
                color = np.random.rand(3)
                while color.std()<0.1:
                    color = np.random.rand(3)
            self.players.append(Player(idx, color, np.array(start[idx])+X) )
        self.race = np.zeros((race.shape[0]+2,race.shape[1]+2), race.dtype)
        self.race[1:-1, 1:-1] = race
        self.end = []
        for pos in end:
            self.end.append((pos[0]+1, pos[1]+1))
        plt.figure()
        self.im = plt.imshow(np.zeros(self.race.shape+(3,)),
                             **plot_dic)

    def next_pos(self, idx):
        """
        returns a list of possible
        moves for player nb idx
        """
        dirs = [X, -X, Y, -Y, 0*X]
        player = self.players[idx]
        return [tuple(player.p + player.v + d)
                for d in dirs]

    def available(self, idx, pos):
        """
        tests if pos is available
        """
        for i in xrange(self.n):
            if idx != i and \
                    pos == tuple(self.players[i].p):
                return False
        try:
            ok = self.race[pos]
        except IndexError, err:
            ok = False
        if pos[0]<0 or pos[1]<0:
            ok = False
        return ok

    def path(self, pos, aim):
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

    def manhattan_dist(self, p1, p2):
        return abs(p2[0]-p1[0]) + abs(p2[1]-p1[1])

    def insert(self, p1, p2):
        assert self.manhattan_dist(p1, p2) == 2, 'positions too far away'
        return [(p1[0], p2[1]), (p2[0], p1[1])]

    def next_move(self, idx):
        """
        asks what to do next
        """
        self.image(idx)
        tmp = raw_input(
            '\nnext move? (use numpad or w-a-s-d-x) ')
        if tmp in UP:
            return -X
        elif tmp in LEFT:
            return -Y
        elif tmp in RIGHT:
            return Y
        elif tmp in DOWN:
            return X
        elif tmp in STOP:
            return O
        elif tmp == QUIT:
            sys.exit()
        else:
            return self.next_move(pl)

    def jitter(self):
        """
        returns random direction with
        probability P_JITTER (else 0)
        """
        if np.random.rand() < 1-P_JITTER:
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

    def rec_pos(self):
        """
        records all positions
        """
        for player in self.players:
            player.rec_position()

    def turn(self, idx):
        """
        one turn
        """
        t = time()
        dv = self.next_move(idx)
        if (time()-t) > PLAY_TIME:
            dv = self.jitter()
        self.players[idx].v += dv
        return self.move(idx)

    def move(self, idx):
        """
        one move
        returns False if anyone has won
        """
        running = True
        player = self.players[idx]
        start = player.p
        aim = start + player.v
        path = self.path(start, aim)
        prev_pos = path.pop(0)
        for pos in path:
            # skip a corner
            if self.manhattan_dist(pos, prev_pos) > 1:
                ways = self.insert(pos, prev_pos)
                if not self.available(idx, ways[0]) and\
                         not self.available(idx, ways[1]):
                    np.random.shuffle(ways)
                    running *= self.crash(idx, ways[0], prev_pos)
                    break
                else:
                    for pos2 in ways:
                        if self.win_check(idx, pos2):
                            running = False
                            prev_pos = pos
                            break
            # crash
            if not self.available(idx, pos):
                running *= self.crash(idx, pos, prev_pos)
                break
            # win ?
            if self.win_check(idx, pos):
                running = False
                prev_pos = pos
                break
            # advance
            prev_pos = pos
        player.p = prev_pos
        self.image()
        return running

    def win_check(self, idx, pos):
        """
        returns False if on the winning line
        """
        win = False
        if pos in self.end:
            print 'player %i wins !'%idx
            self.image()
            win = True
        return win
    
    def crash(self, idx, pos, prev_pos):
        """
        crash
        """
        running = True
        print 'you crashed...'
        player = self.players[idx]
        v_crash = player.v.copy()
        v = abs(v_crash).sum()
        player.play = np.ceil(
            -(1+v)+np.sqrt(1+2*v*(v+1)))
        player.v *= 0
        player.p = prev_pos
        self.image(crash = idx)
        for plr in xrange(self.n):
            if tuple(self.players[plr].p) == pos:
                player = self.players[plr]
                player.v = (0.5*v_crash).round()
                running = self.move(plr)
                player.v = self.rand_dir()
                break
        return running    

    def image(self, idx = None, crash = None, hist = None):
        # color def
        red = np.array([0.7, 0, 0])
        yellow = np.array([0.9, 0.8, 0])
        unif = np.ones(3)
        white = 0.8*unif
        grey = 0.4*unif
        # Image
        im = grey*np.ones(self.race.shape+(3,))
        # race
        im[self.race] = white
        # history
        if hist:
            for plr in xrange(self.n):
                player = self.players[plr]
                for pos in player.history[:hist]:
                    im[pos] = player.color
        # players
        for plr in xrange(self.n):
            player = self.players[plr]
            im[tuple(player.p)] = player.color
        # player turn
        if idx >= 0:
            color = self.players[idx].color
            im[[0,-1]] = color
            im[:, [0,-1]] = color
            # move
            for pos in self.next_pos(idx):
                if pos[0]>-1 and pos[1]>-1:
                    try:
                        im[pos] *= 0.3
                        im[pos] += red
                    except IndexError: pass
        # explosion
        if crash >= 0:
            size = int(self.players[crash].play)
            x,y = tuple(self.players[crash].p)
            it = product(range(x-size,x+size+1),
                         range(y-size,y+size+1))
            for a,b in it:
                if (np.sqrt((a-x)**2+(b-y)**2))<=size:
                    try:
                        im[a,b] = yellow
                    except IndexError: pass
        self.im.set_data(im)
        plt.draw()
        sleep(0.3)

    def replay(self):
        for i in xrange(len(self.players[0].history)+1):
            self.image(hist = i)

    def run(self):
        running = True
        while running:
            for idx in xrange(self.n):
                player = self.players[idx]
                if not running: break
                if not player.play:
                    print '- - - - - - - - - - -\nplayer %i turn' \
                        %idx
                    running = self.turn(idx)
                else:
                    print '- - - - - - - - - - -'
                    print 'player %i misses %i turn' \
                        %(idx, player.play)
                    player.play -= 1
            self.rec_pos()
        for idx in xrange(self.n):
            self.players[idx].reset_pos()
        while 1:
            tmp = raw_input('for no replay, press "q"')
            if tmp == 'q':
                sys.exit()
            self.replay()

#################
###   races   ###
#################

# races

def straight(size, width):
    race = np.ones((size,width), bool)
    start = [(0,i) for i in range(width)]
    end = [(size-1,i) for i in range(width)]
    return (race, start, end)

def s_shape(size, width):
    race = np.zeros((size+2*width,5*width), bool)
    race[:,0:width] = True
    race[:,2*width:3*width] = True
    race[:,4*width:5*width] = True
    race[-width:,width:2*width] = True
    race[:width,3*width:4*width] = True
    start = [(0,i) for i in range(0,width)]
    end = [(size+2*width-1,i) for i in range(4*width,5*width)]
    return (race, start, end)

# obstacles

def obstacles(race, prob):
    return ((np.random.rand(*race.shape)<prob)*race).astype(int)

sgm = 2.
gss_coef = 1/np.sqrt(2*np.pi)/sgm
def filter2D(race, start, obstacles, prob,
             func = lambda x: np.exp(-0.5*x**2/sgm**2)*gss_coef):
    for st in start:
        obstacles[st] = -2
    m = np.zeros(race.shape, float)
    X, Y = race.shape
    it = product(xrange(2*X), xrange(2*Y))
    for x,y in it:
        coef = func(np.sqrt((x-X)**2 + (y-Y)**2))
        if coef>1e-10:
            m[max(0,x-X):min(x,X), max(0,y-Y):min(y,Y)] += coef *\
                obstacles[max(0,X-x): min(2*X-x,X),
                          max(0,Y-y): min(2*Y-y,Y)]
    rand = np.random.rand(*m.shape)
    rand[-race] = 2
    final = rand < prob*m
    plt.figure()
    plt.subplot(221)
    plt.imshow(obstacles)
    plt.subplot(222)
    plt.imshow(final)
    plt.subplot(212)
    plt.imshow(m)
    plt.colorbar()
    return final

################
###   main   ###
################

def usage():
    print '-n # players'
    print '-r race name (straight, s_shape)'
    print '-s race size'
    print '-w race width'
    print '-o obstacles sparseness'
    print '-f filter threshold'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hn:r:s:w:o:f:',
                                   ['help'])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print "option -a not recognized"
        usage()
        sys.exit(2)
    n = 2
    size = 50
    width = 25
    prob = 0
    filt = 0
    race = 'straight'
    for o, a in opts:
        if o == '-n':
            n = int(a)
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        elif o == '-s':
            size = int(a)
        elif o == '-w':
            width = int(a)
        elif o == '-o':
            prob = float(a)
        elif o == '-r':
            race = a
        elif o == '-f':
            if not prob: prob = 0.01
            filt = float(a)
        else:
            assert False, 'unhandled option'
    exec('race = '+race+'(size, width)')
    if prob:
        obs = obstacles(race[0], prob)
    if filt:
        obs = filter2D(race[0], race[1], obs, filt)
    if prob:
        race = (race[0]-obs, race[1], race[2])
    print '%i players' %n
    race = Race(n, *race)
    race.run()

if __name__ == '__main__':
    main()
