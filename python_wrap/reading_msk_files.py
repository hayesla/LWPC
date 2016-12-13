import numpy as np
from sunpy import lightcurve as lc
import matplotlib.pyplot as plt
from pandas import *
from sunpy.time import parse_time
basetime = parse_time('2016-07-23 00:00')
import datetime
import os

# directory where files are stored
#dir_path = '/home/laura/QPP/sid/july_event/VLF_Study/MSK_data/files_rus/'

#----------------------------------------------------#
# function to read in data and output name of station,
# time frame, pandas Series for amplitude  and phase 
# input is name of file
#----------------------------------------------------#
def read_files(name):

	#f = open(os.path.join(dir_path,name))
	f = open(name)	
	ff = f.readlines()


	data = []
	for i in range(0, len(ff)):
		if ff[i][0] != '%':
			data.append(ff[i].split())
	data = np.array(data)

	t = data[:,0]
	a = data[:,1]
	p = data[:,2]


	def floatify(xx):
		tt = []
		for i in range(0, len(xx)):
			t = np.float(xx[i])
			tt.append(t)
		return tt

	t= np.array(floatify(t))
	a = np.array(floatify(a))
	ph = np.array(floatify(p))
	#p = np.unwrap(ph)

	new_time = []
	for i in range(0, len(t)):
		ty = basetime + datetime.timedelta(seconds = t[i])
		new_time.append(ty)


	amp = Series(a, index = new_time)
	pha = Series(p, index = new_time)
	
	#amp = amp.truncate(t_start, t_end)
	#pha = pha.truncate(t_start, t_end)

	return name[0:3],new_time, amp, pha

