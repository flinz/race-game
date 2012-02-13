import numpy as np
import getopt, sys
from copy import copy

from elements import Player, Missile, Ghost
from circuit import Circuit
from graphics import Graphics
from geometry import *

from lorric_agent import RandDriver

############## race #####################

COLOR_ORDER = [
    'blue',
    'green',
    'red',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'white'
    ]

COLOR_GHOST = [
    'light_blue',
    'light_green',
    'light_red',
    'light_cyan',
    'light_magenta',
    'light_yellow'
    ]

class Race:

    def __init__(self, n, ais, shape, length, width, name):
        """
        n: # players
        """
        self.t = 0
        self.n = n + len(ais)
        self.playing = []
        self.done = []
        self.arrived_players = 0
        self.ghosts = []
        self.circuit = Circuit(shape, length, width, name)
        starting_blocks = copy(self.circuit.start)
        np.random.shuffle(starting_blocks)
        for idx in xrange(n):
            color = COLOR_ORDER[idx]
            position = np.array(starting_blocks[idx], int)
            # color = np.random.rand(3)
            # while color.std()<0.1:
            #     color = np.random.rand(3)
            self.playing.append(
                Player(self, color, position) )
        idx = n
        for ai in ais:
            color = COLOR_ORDER[idx]
            position = np.array(starting_blocks[idx], int)
            self.playing.append(
                ai(self, color, position) )
            idx += 1
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
        for ghost in self.ghosts:
            ghost.init_pos()
        self.fill_history()

    def fill_history(self):
        for element in self.done + self.ghosts:
            element.fill_history()
            
    def rec_pos(self):
        """
        records all positions
        """
        for element in self.playing:
            element.rec_position()

    def add_missile(self, player):
        self.i += 1
        idx = len(self.playing)
        missile = Missile(self, player.p,
                          player.v)
        self.playing.insert(
            self.playing.index(player),
            missile)
        self.move(missile)

    def add_ghost(self, name):
        idx = len(self.ghosts)
        ghost = Ghost(self, idx,
                      COLOR_GHOST[idx],
                      name)
        self.ghosts.append(ghost)

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
            # at a corner
            if manhattan_dist(pos, prev_pos) > 1 and\
                    self.corner(element, pos, prev_pos):
                break
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

    def corner(self, element, pos, prev_pos):
        ways = insert(pos, prev_pos)
        ways = [(way, self.available(element, way) )
                for way in ways]
        np.random.shuffle(ways)
        hit = False
        # both blocked for driver
        if element.avoid and ways[0][1] and ways[1][1]:
            hit = True
            where, what = ways[0]
        # 1st blocked for missile
        elif (not element.avoid) and ways[0][1]:
            hit = True
            where, what = ways[0]
        # 2nd blocked for missile
        elif (not element.avoid) and ways[1][1]:
            hit = True
            where, what = ways[1]
        # hit (if any of the three above)
        if hit:
            self.hit(element, where, what)
            return True
        # win check in any of the two
        for way, _ in ways:
            if self.circuit.win_check(way):
                element.win(way)
                return True
        return False

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
        self.t += 1
        print 'turn %i'%self.t
        self.i = 0
        while self.i < len(self.playing):
            element = self.playing[self.i]
            if not element.play:
                print '- - - - - - - - - - -'
                element.turn()
                self.move(element)
            else:
                print '- - - - - - - - - - -'
                print 'you miss %i turn'%element.play
                element.play -= 1
            self.i += 1
        for ghost in self.ghosts:
            ghost.turn()
        self.rec_pos()
        self.graphics.draw()
            
    def run(self):
        self.graphics.draw()
        tmp = raw_input('press RETURN to start, Q to quit ')
        if tmp == 'q':
            sys.exit()
        while self.arrived_players < min(3, max(self.n-1, 1)):
            self.turn()
        self.close()
        while 1:
            tmp = raw_input('press Q to quit, S to save circuit ')
            if len(tmp) and tmp in 'Qq':
                sys.exit()
            if len(tmp) and tmp in 'Ss':
                name = raw_input('\nname ? ')
                self.circuit.write(name)
                sys.exit()
            self.replay()

    def replay(self):
        for i in xrange(len(self.done[0].history)+1):
            self.graphics.draw(history = i)

############## main ##############

def usage():
    print '\n-n # players'
    print '-c circuit file name to load'
    print '-r race type (straight, s_shape)'
    print '-l race length'
    print '-w race width'
    print '-o obstacles density'
    print '-s filter sigma'
    print '-f filter threshold'
    print '-g ghost names (separate with coma)'
    print '-a add AIs'
    print '\n--help   prints this'
    print '--filter automatic parameter set'
    print '         density = 0.01, sigma = 1., threshold = 5.\n'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:n:r:l:w:o:f:s:g:a',
                                   ['help', 'filter'])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print "option -a not recognized"
        usage()
        sys.exit(2)
    n = 2
    length = 20
    width = 20
    density = 0
    sigma = 0
    threshold = 0
    shape = 'straight'
    name = None
    ghosts = []
    ais = []
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
        elif o == '-g':
            ghosts = a.split(',')
        elif o == '-a':
            ais = add_ai()
        elif o == '-c':
            name = a
        elif o == '--filter':
            density = 0.01
            sigma = 1.
            threshold = 5.
        else:
            assert False, 'unhandled option'
    race = Race(n, ais, shape, length, width, name)
    for name in ghosts:
        race.add_ghost(name)
    if density:
        race.circuit.add_obstacles(density)
    if threshold or sigma:
        race.circuit.filter_obstacles(threshold, sigma) # , True)
    race.run()

def add_ai():
    n = raw_input('\nHow many ai ? ')
    ais = []
    for _ in xrange(int(n) ):
        ai_class = raw_input('\nAI class : ')
        exec('ais.append('+ai_class+')')
    return ais

if __name__ == '__main__':
    main()
