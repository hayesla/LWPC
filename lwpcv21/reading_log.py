#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

def main(argv):
    
    file_name = argv[0]
    a = open(file_name)
    lines = a.readlines()
    all_data = []
    for i in range(len(lines)):
        all_data.append(lines[i].split())

    for i in range(len(all_data)):
        try:
            if all_data[i][0] == 'dist':
                start_data = i
            elif all_data[i][0] == 'nc':
                end_data = i
        except:
            print('nada')


    meta = all_data[0:start_data]
    test = all_data[start_data+1: end_data]
    rest = all_data[end_data:]


    #test = []
    #for i in range(len(data)):
    #    test.append(data[i].split())

    test = np.array(test)
    dist1 = []
    dist2 = []
    dist3 = []
    for i in range(1, len(test)):
        dist1.append(test[i][0])
        dist2.append(test[i][3])
        try:
            dist3.append(test[i][6])
        except:
            print('end')
    distance = dist1 + dist2 + dist3
    for i in range(len(distance)):
        distance[i] = float(distance[i])


    amp1 = []
    amp2 = []
    amp3 = []
    for i in range(1, len(test)):
        amp1.append(test[i][1])
        amp2.append(test[i][4])
        try:
            amp3.append(test[i][7])
        except:
            print('end')
    amp = amp1 + amp2 + amp3
    for i in range(len(amp)):
        amp[i] = float(amp[i])

    pha1 = []
    pha2 = []
    pha3 = []
    for i in range(1, len(test)):
        pha1.append(test[i][2])
        pha2.append(test[i][5])
        try:
            pha3.append(test[i][8])
        except:
            print('end')
    pha = pha1 + pha2 + pha3
    for i in range(len(pha)):
        pha[i] = float(pha[i])

    fig, ax = plt.subplots(2, sharex = True)
    ax[0].plot(distance[0:-1], pha[0:-1], label = 'Phase')
    ax[0].legend()
    ax[0].set_xlabel('Distance (km)')
    ax[1].plot(distance[0:-1], amp[0:-1], label = 'Amplitude', color = 'r')
    ax[1].legend()
    ax[1].set_xlabel('Distance (km)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
