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
	i_d = 'test_file_dho'
	freq = '23.400'
	tx_lat = '53.1'
	tx_lon = '7.6'
	rx_lat = '55.267'
	rx_lon = '-3.1757'
	power = '1.000e+03'
	date = 'Jul/24/2016'
	hhmm = '%02d:%02d' %(mins/60, mins%60)
	
	#creating inp file for the lwpc code
	f = open('test_file_dho.inp', 'w')
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
	input_file = 'test_file_dho'
	run_command = './test ' + input_file
	run_lwpc = subprocess.call(run_command, shell = True)
	print hhmm

	pha, amp, dist = read_lwpc_log('test_file_dho.log')
	wr = open('24_test_dho.dat', 'a')
	wr.write(hhmm + ' '  + str(pha[-1]) + ' ' + str(amp[-1]) + ' ' + str(dist[-1]) + '\n')
	wr.close()

	mins = mins+10

res = np.loadtxt('24_test_dho.dat', dtype = 'str')
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



