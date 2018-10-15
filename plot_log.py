"""plot general log file according to given indexes"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def plot(filename):

    data = []

    with open(filename, 'r') as fin:
        for line in fin.readlines():
            items = line.split('\t')

            row = []
            for item in items[1:]:
                t = eval(item.split(':')[1])
                if isinstance(t, list):
                    for x in t:
                        row.append(x)
                else:
                    row.append(t)
            if len(row) > 0:
                data.append(row)

    data = np.array(data)
    print(data.shape)
    num = 100
    #print(data[-num:, 0].mean())
    print(data[-num:, 1].mean())
    print(data[-num:, 2].mean())

    for index in [1, 2]:
        index = int(index)
        plt.plot(moving_average(data[:, index], 100), label=filename+'agent' if index == 1 else filename+'agent_strong')

    #plt.show()

if __name__=='__main__':
    import sys
    plot('gathering_3-2.log')
    plot('gathering_3-2_1.log')

    print('saving ...')
    plt.legend()
    plt.savefig('tmp.png')
