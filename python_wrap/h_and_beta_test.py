import numpy as np
import subprocess
import os
from reading_log import read_lwpc_log
import datetime
import matplotlib.pyplot as plt
plt.ion()


loopy_h = np.arange(60, 80, 0.1)
loopy_b = np.arange(0.3, 0.5, 0.001)
for i in loopy_h:
	for j in loopy_b:
		workdir='/home/laura/lwpc_test/lwpc/python_wrap'
		i_d = 'h_and_beta'
		freq = '24.000'
		tx_lat = '44.633'
		tx_lon = '67.283'
		rx_lat = '53.091'
		rx_lon = '-7.913'
		power = '1.000e+03'
		date = 'Jul/23/2016'
		#hhmm = '%02d:%02d' %(mins/60, mins%60)
		
		#creating inp file for the lwpc code
		f = open('h_and_beta.inp', 'w')
		f.write('file-mds '+ workdir +'\n')
		f.write('file-lwf '+ workdir +'\n')
		f.write('tx '+ i_d +'\n')
		f.write('tx-data '+ i_d + ' ' + freq + ' ' + tx_lat + ' ' + tx_lon +' '+power + ' 0 0 0'+'\n')
		f.write('ionosphere homogeneous exponential '+str(j) + ' '+ str(i) + '\n')
		f.write('receivers ' + rx_lat +' '+ rx_lon + '\n')
		f.write('print-lwf 1 \n')
		f.write('start \n')
		f.write('quit \n')
		f.close()
	
		input_file = 'h_and_beta'
		run_command = './test ' + input_file
		run_lwpc = subprocess.call(run_command, shell = True)
		print 'h_prime equals' +str(i)+ 'and beta equals ' + str(j)

		pha, amp, dist = read_lwpc_log('h_and_beta.log')
		wr = open('h_and_beta_full.dat', 'a')
		wr.write(str(i) + ' ' + str(j) + ' '  + str(pha[-1]) + ' ' + str(amp[-1]) + ' ' + str(dist[-1]) + '\n')
		wr.close()
	
res = np.loadtxt('h_and_beta_full.dat')
h_prime = res[:,0]
phase = res[:,1]
amp = res[:,2]

