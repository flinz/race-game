import sys
import matplotlib.pyplot as plt

def main(name):
    f = open(name+'times', 'r')
    times = []
    for line in f:
        times.append(int(line) )
    plt.plot(times)
    plt.show()

if __name__ == '__main__':
    name = sys.argv[1]
    main(name)
