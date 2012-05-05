import sys

import race
from lorric_agent import NNDriver

N = 10000

def main(name):
    f = open(name+'times', 'w')

    rc = race.Race(0, 's', 15, 15, False)
    nn = NNDriver(rc, name)
    rc.add_ai(nn)
    for i in range(N):
        print 'run nb %i'%i
        rc.run()
        f.write('%i\n'%rc.t)
        if not i%(N/10):
            nn.write(name+'_%i'%i)
    nn.save(name)

if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
