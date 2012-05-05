import sys
import matplotlib.pyplot as plt
import numpy as np

import lorric_agent as la

length = 20
width = 20
p_len = 86
v_len = 45
a_len = 5

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

def what_v(direction):
    idx = []
    if direction == 'null':
        idx.append(0)
    elif direction == 'down':
        idx += range(3, 6)
        idx += range(14, 20)
        idx += range(33, 38)
    elif direction == 'up':
        idx += [1]
        idx += range(7, 10)
        idx += range(24, 30)
        idx += range(41, 45)
    elif direction == 'left':
        idx += range(1, 4)
        idx += range(9, 15)
        idx += range(29, 34)
    elif direction == 'right':
        idx += range(5, 8)
        idx += range(19, 25)
        idx += range(37, 42)
    return idx

def average(cx, v_idx):
    return cx[:,:, v_idx].mean(-1)

def xy(p_field):
    X = np.zeros(p_len)
    Y = np.zeros(p_len)
    for p in xrange(p_len):
        x0, x1, y0, y1 = p_field[p]
        X[p] = x0 + (x1-x0)//2
        Y[p] = y0 + (y1-y0)//2
    return X, Y

def load(name):
    cx = np.zeros(
        (a_len, p_len, v_len) )
    try:
        name += '.nna'
        print 'opening NNAgent '+name
        f = open(name, 'r')
        assert f.readline() == '# NN agent\n', 'wrong format'
        for line in f:
            line = line.split(' ')
            i, jk = int(line[0]), int(line[1])
            j, k = divmod(jk, v_len)
            w = float(line[2])
            cx[i, j, k] = w
        f.close()
        print 'closed'
    except IOError:
        print 'no such file...'
        sys.exit()
    return cx

# cx = 0, down, right, up, left

rmax = 10

def arrow(cx):
    actx = np.zeros(p_len)
    acty = np.zeros(p_len)
    for p in xrange(p_len):
        act = cx[:,p]
        coef = (act[0]+rmax+1)/(rmax+1)
        actx[p] = coef*(act[1]-act[3])
        acty[p] = coef*(act[2]-act[4])
    return actx, acty

directions = ['null', 'down', 'up', 'left', 'right']

def main(name):
    pf = la.p_field(length, width)
    X, Y = xy(pf)
    cx = load(name)
    dic = {}
    for d in directions:
        dic[d] = average(cx, what_v(d) )
    for drc,cxx in dic.iteritems():
        plt.figure()
        plt.title(drc)
        Ax, Ay = arrow(cxx)
        plt.quiver(Y, X, Ay, Ax)
        plt.show()

if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
