import numpy as np
import subprocess
import os
from reading_log import read_lwpc_log
import datetime
import matplotlib.pyplot as plt
plt.ion()



mins = 0
while mins < 1440:
#parameters to do into lwpc inp file
	workdir='/home/laura/lwpc_test/lwpc/python_wrap'
	i_d = 'test_file'
	freq = '24.000'
	tx_lat = '44.633'
	tx_lon = '67.283'
	rx_lat = '53.091'
	rx_lon = '-7.913'
	power = '1.000e+03'
	date = 'Jul/23/2016'
	hhmm = '%02d:%02d' %(mins/60, mins%60)
	
	#creating inp file for the lwpc code
	f = open('test_file.inp', 'w')
	f.write('file-mds '+ workdir +'\n')
	f.write('file-lwf '+ workdir +'\n')
	f.write('tx '+ i_d +'\n')
	f.write('tx-data '+ i_d + ' ' + freq + ' ' + tx_lat + ' ' + tx_lon +' '+power + ' 0 0 0'+'\n')
	f.write('ionosphere lwpm ' + date + ' '+ hhmm + '\n')
	#f.write('ionosphere homogeneous exponential 0.3 76 \n')
	f.write('receivers ' + rx_lat +' '+ rx_lon + '\n')
	f.write('print-lwf 1 \n')
	f.write('start \n')
	f.write('quit \n')
	f.close()
	
	#will run LWPC given the filename
	input_file = 'test_file'
	run_command = './test ' + input_file
	run_lwpc = subprocess.call(run_command, shell = True)
	print hhmm

	pha, amp, dist = read_lwpc_log('test_file.log')
	wr = open('23_test.dat', 'a')
	wr.write(hhmm + ' '  + str(pha[-1]) + ' ' + str(amp[-1]) + ' ' + str(dist[-1]) + '\n')
	wr.close()

	mins = mins+10

res = np.loadtxt('23_test.dat', dtype = 'str')
res = np.array(res)
tim = []
for i in range(len(res)):
    tim.append(datetime.datetime.strptime(date+' '+res[:,0][i], '%b/%d/%Y %H:%M'))
pha = res[:,1]
amp = res[:,2]

fig, ax = plt.subplots(figsize = (10,5))
ax.plot(tim, pha, label = 'Phase', color = 'g')
ax.set_title('LWPC ' + date + ' ' + freq + ' kHz TX: '+ tx_lat  +','+tx_lon + ' RX: ' + rx_lat + ','+ rx_lon)
ax.set_ylabel('Phase (degrees)')
ax.legend(loc = 'upper left')
ax2 = ax.twinx()
ax2.plot(tim, amp, label = 'Amplitude', color = 'r')
ax2.set_ylabel('Amplitude dBuV/m')
ax2.set_xlabel(date + ' hh:mm')
ax2.legend(loc = 'upper right')



