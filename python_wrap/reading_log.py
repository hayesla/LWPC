#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys



#def main(argv):
def read_lwpc_log(file_name):
    
    #file_name = argv[0]
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
            pass


    meta = all_data[0:start_data]
    test = all_data[start_data+1: end_data]
    rest = all_data[end_data:]

    #takes care of the invalid literal for amp and phase sticking together

    check = True
    while check:
        check = False
        for i in range(len(test)):
            if len(test[i]) < 9:
                for j in range(len(test[i])):
                    if len(test[i][j])>12:
                        ll = test[i][j].split('-')
                        if len(ll) > 2:
                            test[i][j] = '-'+ll[1]
                            test[i].insert(j+1, ll[2])
                        else:
                            test[i][j] = ll[0]
                            test[i].insert(j+1, '-'+ll[1])
        for i in range(len(test)):
            for j in range(len(test[i])):
                if len(test[i][j]) > 12:   
                    check = True


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
            pass
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
            pass
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
            pass
    pha = pha1 + pha2 + pha3
    for i in range(len(pha)):
        pha[i] = float(pha[i])
    return distance, amp, pha

