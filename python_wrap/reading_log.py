#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set_context('paper', font_scale = 1.5)
sns.set_style('ticks',{'xtick.direction':'in','ytick.direction':'in'})


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

    wah = True
    while wah:
        wah = False
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
                    wah = True

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
    return pha, amp, distance
    '''
    fig, ax = plt.subplots(2, sharex = True)
    if file_name == 'gqd.log':
    ax[0].set_ylim(0,500)ls
    #ax[1].set_ylim(50, 110)
        ax[0].set_xlim(0, 2000)
    if file_name == 'naaa.log':
        ax[0].set_xlim(0, 6550)
        ax[0].set_ylim(-100,400)
        ax[1].set_ylim(50, 120)
    ax[0].plot(distance[0:-1], pha[0:-1], label = 'Phase')
    ax[0].legend()
    ax[0].set_xlabel('Distance (km)')
    ax[1].plot(distance[0:-1], amp[0:-1], label = 'Amplitude', color = 'r')
    ax[1].legend()
    ax[1].set_xlabel('Distance (km)')
    plt.tight_layout()
   # plt.savefig('test1.png')
    plt.show()'''
    

#if __name__ == "__main__":
#    main(sys.argv[1:])

