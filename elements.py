import sys
import numpy as np
from time import sleep, time
from geometry import X, Y, O

UP = '8w'
LEFT = '4a'
RIGHT = '6d'
DOWN = '2x'
STOP = '5s'
QUIT = 'quit'
MISSILE = 'm'

############### base class ##############

class Element:

    gh_opening = '# ghost format copyright Ziegler 2012\n'

    def __init__(self, race, color,
                 position = None, velocity = None):
        self.register(race)
        self.color = color
        self.init_pv(position, velocity)
        self.init_history()
        self.avoid = False
        self.miss = 0

    def init_pv(self, position = None,
                velocity = None):
        if position:
            self.p = np.array(position, int)
        else:
            self.p = O.copy()
        if velocity:
            self.v = np.array(velocity, int)
        else:
            self.v = O.copy()

    def init_color(self, color):
        self.color = color

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
        if self.miss:
            self.miss -= 1
        return self.miss

    def win(self, pos):
        self.p = np.array(pos, int)
        self.end()

    def end(self):
        self.rec_position()
        self.race.out(self)

    def write(self, name):
        print 'writing ghost '+name
        f = open(name + '.ghs', 'w')
        f.write(self.gh_opening)
        for pos in self.history:
            if len(pos):
                f.write('%i %i\n'%pos)
        f.close()
        print 'done'

############# Car #####################

class Car(Element):

    v_in = 0.5
    v_into = 0.5

    def __init__(self, race):
        Element.__init__(self, race, 'white')   
        self.avoid = True

    def hit(self, what):
        """
        hits wall or other element
        """
        if self.race.display:
            print 'you crashed...'
        if what.__class__ == Player or what == 'wall':
            v = abs(self.v).sum()
            self.miss = int(np.ceil(
                    -(1+v)+np.sqrt(1+2*v*(v+1) ) ) )
            self.v = O.copy()
        elif what.__class__ == Missile:
            self.be_hit(what)

    def be_hit(self, element, v):
        if element.__class__ == Player:
            self.v = (self.v_in*v +
                      self.v_into*self.v).round()
            self.race.move(self)
        if element.__class__ == Missile:
            pass
        # if hasn't crashed
        if not self.miss:
            self.v = self.rand_dir()
        
    def rand_dir(self):
        """
        returns random direction
        """
        sign = (2*np.random.randint(0,2)-1)
        direction = X.copy() + (Y.copy()-X.copy() )*(
            np.random.rand()<0.5)
        return sign*direction

    def win(self, pos):
        Element.win(self, pos)
        self.race.arrived_players += 1

############### players #################

class Player(Car):

    missile = 10
    play_time = 10
    p_jitter = 0.05

    def __init__(self, race):
        Car.__init__(self, race)   
        self.missile = self.missile

    def turn(self):
        if self.miss:
            self.miss -= 1
        else:
            t = time()
            dv, missile = self.next_move()
            if (time()-t) > self.play_time:
                dv = self.jitter()
            if missile and self.missile > 0:
                self.race.add_missile(self)
                self.missile -= 1
            self.v += dv
        return self.miss

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
        probability p_jitter (else 0)
        """
        if np.random.rand() < 1-self.p_jitter:
            direction = O.copy()
        else:
            direction = self.rand_dir()
        return direction

    def win(self, pos):
        Car.win(self, pos)
        if self.race.display:
            save = raw_input('\nsave ghost [y/N]? ')
            if save == 'y':
                name = raw_input('name ? ')
                self.write(name)

############### ghosts #################

class Ghost(Element):

    opening = '# ghost format copyright Ziegler 2012\n'

    def __init__(self, race, color, name):
        Element.__init__(self, race, color)   
        self.read(name)
        self.p = np.array(self.history[0])

    def init_pv(self, *args):
        pass

    def init_history(self):
        self.history = []

    def turn(self):
        try:
            self.p = np.array(self.history[self.race.t])
        except IndexError, err:
            pass

    def read(self, name):
        name += '.ghs'
        print 'opening', name, '...'
        f = open(name, 'r')
        assert f.readline() == self.gh_opening, 'wrong format'
        for line in f:
            line = line.split(' ')
            self.history.append( (int(line[0]), int(line[1]) ) )
        f.close()
        print 'closed'

############## missile ##################

class Missile(Element):

    V = 10
    SIZE = 3

    def __init__(self, race, car_position,
                 car_velocity):
        vx, vy = tuple(car_velocity)
        v = np.sqrt( (car_velocity**2).sum() )
        if v == 0:
            vx = np.random.rand()
            vy = np.random.rand()
            v = np.sqrt(vx**2+vy**2)
        velocity = (round(vx*self.V/v),
                    round(vy*self.V/v) )
        Element.__init__(self, race, 'black',
                         tuple(car_position), velocity)

    def hit(self, what):
        self.race.i -= 1
        self.end()

    def be_hit(self, element, v):
        self.end()

    def end(self):
        Element.end(self)
