import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from itertools import product
from geometry import X, Y, O

plt.ion()
plot_dic={'interpolation':'nearest'}

def replace_tuple(tup):
    return tup[0]+1, tup[1]+1

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
    'light_blue': (0.6, 0.6, 0.8),
    'light_green': (0.6, 0.7, 0.6),
    'light_red': (0.8, 0.6, 0.6),
    'light_cyan': (0.6, 0.7, 0.7),
    'light_magenta': (0.7, 0.6, 0.7),
    'light_yellow': (0.7, 0.7, 0.6)
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
        for element in self.race.done +\
                self.race.junk + self.race.ghosts:
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
        for element in self.race.playing + self.race.ghosts:
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
