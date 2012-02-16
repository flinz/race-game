import numpy as np
from copy import copy

from elements import Player, Missile, Ghost
from circuit import Circuit
from graphics import Graphics
from geometry import *

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

    t_max = 1000

    def __init__(self, n, shape, length, width,
                 display = True):
        """
        n: # players
        """
        self.t = 0
        self.n = n
        self.playing = []
        self.done = []
        self.ghosts = []
        self.circuit = Circuit(shape, length, width)
        for idx in xrange(n):
            self.add_player()
        self.display = display
        if display:
            self.graphics = Graphics(self, self.circuit)

    def init(self):
        self.t = 0
        self.arrived_players = 0
        self.playing += self.done
        self.done = []
        self.init_pv()

    def init_pv(self):
        starting_blocks = copy(self.circuit.start)
        np.random.shuffle(starting_blocks)
        for i in range(len(self.playing) ):
            element = self.playing[i]
            element.init_pv(starting_blocks[i])
            element.init_history()

    def init_color(self):
        for i in range(len(self.playing) ):
            element = self.playing[i]
            element.init_color(COLOR_ORDER[i])

    def out(self, element):
        try:
            self.playing.remove(element)
        except ValueError, err:
            pass
        if element not in self.done:
            self.done.append(element)

    def close(self):
        for element in self.playing:
            element.end()
        for ghost in self.ghosts:
            ghost.init_pv()
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
        ghost = Ghost(self, COLOR_GHOST[idx],
                      name)
        self.ghosts.append(ghost)

    def add_ai(self, ai):
        self.playing.append(ai)

    def add_player(self):
        self.playing.append(Player(self) )

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
        if self.display:
            if what.__class__ == Missile or\
                    element.__class__ == Missile:
                size = Missile.SIZE
            else:
                size = element.miss
            self.graphics.draw(
                explosion = (pos, size) )
        # to what
        if what != 'wall':
            what.be_hit(element, v)

    def turn(self):
        self.t += 1
        if self.display:
            print 'TURN %i'%self.t
            print '- - - - - - - - - - -'
        self.i = 0
        while self.i < len(self.playing):
            element = self.playing[self.i]
            miss = element.turn()
            if not miss:
                self.move(element)
            elif self.display:
                print 'you miss %i turn'%miss
            self.i += 1
        for ghost in self.ghosts:
            ghost.turn()
        self.rec_pos()
        if self.display:
            self.graphics.draw()
            
    def run(self):
        self.init()
        if self.display:
            self.init_color()
            self.graphics.draw()
            tmp = raw_input('press RETURN to start, Q to quit ')
            if tmp == 'q':
                sys.exit()
        while self.arrived_players < min(3, max(self.n-1, 1)) and\
                self.t < self.t_max:
            self.turn()
        self.close()

    def replay(self):
        for i in xrange(len(self.done[0].history)+1):
            self.graphics.draw(history = i)
