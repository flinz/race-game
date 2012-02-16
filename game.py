import getopt, sys

from race import Race

from lorric_agent import RandDriver, NNDriver

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
    n = 2
    length = 15
    width = 15
    density = 0
    sigma = 0
    threshold = 0
    shape = 's'
    ghosts = []
    ais = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:n:r:l:w:o:f:s:g:a',
                                   ['help', 'filter'])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print "option -a not recognized"
        usage()
        sys.exit(2)

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
            ais = True
        elif o == '-c':
            shape = a
        elif o == '--filter':
            density = 0.01
            sigma = 1.
            threshold = 5.
        else:
            assert False, 'unhandled option'

    race = Race(n, shape, length, width)
    for name in ghosts:
        race.add_ghost(name)
    if ais:
        add_ai(race)
    if density:
        race.circuit.add_obstacles(density)
    if threshold or sigma:
        race.circuit.filter_obstacles(threshold, sigma) # , True)
    race.run()
    ask(race)

def ask(race):
    while 1:
        tmp = raw_input('R for replay, '+
                        'M for one more (just one..), '+
                        'Q to quit, '+
                        'S to save circuit ')
        if len(tmp) and tmp in 'Rr':
            race.replay()
        if len(tmp) and tmp in 'Mm':
            race.run()
        if len(tmp) and tmp in 'Qq':
            sys.exit()
        if len(tmp) and tmp in 'Ss':
            name = raw_input('\nname ? ')
            race.circuit.write(name)
            sys.exit()

def add_ai(race):
    n = raw_input('\nHow many ai ? ')
    for _ in xrange(int(n) ):
        ai_class = raw_input('\nAI class: ')
        name = raw_input('name: ')
        exec('ai = '+ai_class+'(race,\''+name+'\')')
        race.add_ai(ai)

if __name__ == '__main__':
    main()
