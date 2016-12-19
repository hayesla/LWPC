import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import datetime
from reading_msk_files import read_files
from pandas import Series
from sunpy.time import parse_time
###lwpc data##
'''
res = np.loadtxt('results_23_jul_h_76.dat', dtype = 'str')
date = 'Jul/23/2016'
res = np.array(res)
tim = []
for i in range(len(res)):
    tim.append(datetime.datetime.strptime(date+' '+res[:,0][i], '%b/%d/%Y %H:%M'))
pha_lwpc = res[:,1]
amp_lwpc = res[:,2]

lwpc_amp = Series(amp_lwpc, index = tim)
lwpc_pha = Series(pha_lwpc, index = tim)

'''
###msk data ##

name, msk_time, amp, pha = read_files('NAA20160723.txt')

msk_amp = Series(amp, index = msk_time)+107
msk_pha = Series(pha, index = msk_time)


res = np.loadtxt('h_and_beta_full.dat')
h_prime = res[:,0]
beta = res[:,1]
phase = res[:,2]
amp = res[:,3]
dist = res[:,4]


b_values = np.arange(0.3, 0.5, 0.001)
h_values = np.arange(60, 80, 0.1)

test = np.arange(0, 40000, 200)
new = []
for i in range(len(test)):
	if test[i]!= 39800:
		new.append(amp[test[i]:test[i+1]])
	else:
		new.append(amp[test[i]:])


new = np.array(new)

full_chi = []
for i in range(len(msk_pha)):
	chi = (-new + msk_amp[i])**2
	full_chi.append(chi)





plt.contourf(b_values, h_values, new, cmap = 'magma')
