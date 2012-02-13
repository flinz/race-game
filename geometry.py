import numpy as np

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
