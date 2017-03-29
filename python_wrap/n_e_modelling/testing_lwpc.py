import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import datetime
from reading_msk_files import read_files
from pandas import Series
from sunpy.time import parse_time
from scipy import ndimage
smooth = ndimage.filters.uniform_filter
from sunpy import lightcurve as lc
import matplotlib.dates as dates
import pandas as pd
import seaborn as sns
sns.set_style('ticks',{'xtick.direction':'in','ytick.direction':'in'})
sns.set_context('paper', font_scale=1.8)
from scipy.optimize import curve_fit

###lwpc data##
res = np.loadtxt('24_lwpm_test.dat', dtype = 'str')
date = 'Jul/24/2016'
res = np.array(res)
tim = []
for i in range(len(res)):
    tim.append(datetime.datetime.strptime(date+' '+res[:,0][i], '%b/%d/%Y %H:%M'))
pha_lwpc = res[:,1]
amp_lwpc = res[:,2]

lwpc_amp = Series(amp_lwpc, index = tim)
lwpc_pha = Series(pha_lwpc, index = tim)


###msk data ##

name, msk_time, amp_naa, pha_naa = read_files('NAA120160724.txt', '2016-07-24 00:00')

msk_amp_real = Series(amp_naa, index = msk_time)+107
msk_pha_real = Series(pha_naa, index = msk_time)+45

msk_amp_flare = msk_amp_real.truncate('2016-07-24 11:30','2016-07-24 16:00')
msk_pha_flare = msk_pha_real.truncate('2016-07-24 11:30','2016-07-24 16:00')


peak_t = np.loadtxt('gl_peaks.dat', dtype = 'str')
peak_times = []
for i in range(len(peak_t)):
	peak_times.append(peak_t[i][0] + ' ' +peak_t[i][1])


#goes setup
g = lc.GOESLightCurve.create(msk_amp_flare.index[0], msk_amp_flare.index[-1])
gl = g.data['xrsb']
gs = g.data['xrsa']

#sid_series
def make_sid_series(file_name):
	a = np.loadtxt(file_name, comments = '#', delimiter = ',', dtype = 'str')
	times = []
	data = []
	for i in range(len(a)):
		t = datetime.datetime.strptime(a[i][0], "%Y-%m-%d %H:%M:%S")
		times.append(t)
        
		data.append(a[i][1])
    
	for i in range(len(data)):
		data[i] = float(data[i])
    
    
	sid_full = Series(data, index = times)
	#sid  = sid_full.truncate(t_start, t_end)
	return sid_full

sid_dataa = make_sid_series('BIR_sid_20160724_000000.txt')
sid_dataa = 20*np.log10(sid_dataa+5) - 61 + 107
sid_flare = sid_dataa.truncate('2016-07-24 11:30','2016-07-24 16:00')

s_a = np.array(sid_flare)
s_a[9598: 9623] = np.nan #for 11:30
#s_a[11398:11423] = np.nan #for 11:00
not_nan = np.logical_not(np.isnan(s_a))
indices = np.arange(len(s_a))
new_sa = np.interp(indices, indices[not_nan], s_a[not_nan])
sidy = Series(new_sa, index = sid_flare.index)

#reading hprime and beta values
res = np.loadtxt('test_h_and_beta.dat')
h_prime = res[:,0]
beta = res[:,1]
phase = res[:,2]
amp = res[:,3]
dist = res[:,4]

#for h_and_beta_model.dat
#hh = np.arange(55, 85, 0.1)
#bb = np.arange(0.2, 0.5, 0.001)

#for h_and_beta.dat
#hh = np.arange(60, 80, 1)
#bb = np.arange(0.3, 0.5, 0.01)

hh = np.arange(60, 75.1, 0.1)
bb = np.arange(0.3, 0.5, 0.001)


test = np.arange(0, len(bb)*len(hh), len(bb))
#rows are same height, columns are change in beta
amp_l = []
pha_l = []
for i in range(len(test)):
    if test[i] != test[len(test)-1]:
        amp_l.append(amp[test[i]:test[i+1]])
        pha_l.append(phase[test[i]:test[i+1]])
    else:
        amp_l.append(amp[test[i]:])
        pha_l.append(phase[test[i]:])

amp_l = np.array(amp_l)
pha_l = np.array(pha_l)

##testing mode with chi squared

phase_test = msk_pha_flare.resample('1min', how = 'mean')
amp_test = msk_amp_flare.resample('1min', how = 'mean')
amp_b = sidy.resample('1min', how = 'mean')
gll = gl.resample('1min', how = 'mean')
indexx_phase = []
indexx_amp = [] 
for i in range(len(amp_b)):
    yy = (phase - phase_test[i])**2
    yyy = (amp - amp_b[i])**2
    ind = np.where(yy == np.min(yy))[0][0]
    ind_amp = np.where(yyy == np.min(yyy))[0][0]
    indexx_phase.append(ind)
    indexx_amp.append(ind_amp)

##finding indices of gl peak

datt = []
for i in range(len(peak_times)):
	datt.append(datetime.datetime.strptime(peak_times[i], '%Y-%m-%d %H:%M:%S.%f'))


indd = []
for i in range(len(datt)):       
	b = (datt[i] - amp_b.index).total_seconds()                        
	c = np.where(np.abs(b) == np.min(np.abs(b)))[0][0]
	indd.append(c)



plt.plot(amp_b.index, beta[indexx_amp])


#for i in range(len(test_a)):
def plot_arrays():

    for i in range(len(amp_b)):

        beta_tick = np.arange(0, 201, 50)
        print beta_tick
        beta_label = np.arange(0.3, 0.51, 0.05)
        h_tick = np.arange(0, 200, 50)
        h_label = np.arange(60, 75.1, 5)

        a = (amp - amp_b[i])**2
        test_sol = []
        for j in range(len(test)):
            if test[j]!= test[len(test)-1]:
                test_sol.append(a[test[j]:test[j+1]])
            else:    
                test_sol.append(a[test[j]:])



        fig, ax = plt.subplots(2, figsize = (5, 9))
        ca = ax[0].imshow(test_sol, origin = 'lower', cmap = 'Spectral_r')
       # ax[0].set_xticks(beta_tick)
       # ax[0].set_xticklabels(beta_label)
       # ax[0].set_yticks(h_tick)
       # ax[0].set_yticklabels(h_label)
        ax[0].set_ylabel('H prime (km)')
        ax[0].set_xlabel('Beta (km $^{-1}$')
        ax[0].set_title(str(amp_b.index[i]))
        fig.colorbar(ca, ax = ax[0])

        ax[1].plot(amp_b)
        ax[1].axvline(amp_b.index[i], ls = '--', color = 'k')
        ax[1].set_ylabel('Amplitude dBuV/m')
        ax[1].set_xlabel('Start time '+str(amp_b.index[0])[0:16]+' UT')
        plt.tight_layout()

        if i < 10:
            plt.savefig('/home/laura/lwpc_test/lwpc/python_wrap/n_e_modelling/plots_phase/test000'+str(i)+'.png')
            plt.clf()
        elif i>9 and i <100:
            plt.savefig('/home/laura/lwpc_test/lwpc/python_wrap/n_e_modelling/plots_phase/test00'+str(i)+'.png')
            plt.clf()
        else:
            plt.savefig('/home/laura/lwpc_test/lwpc/python_wrap/n_e_modelling/plots_phase/test0'+str(i)+'.png')
            plt.clf()





#1.43e7 in cm^-3, 1.42e13 in m^3
def n_e(h, h_prime, beta):
    return 1.43e13*np.exp(-0.15*h_prime)*np.exp((beta - 0.15)*(h-h_prime))

#function to plot 3d map of e density of time and h
def plot_elec_den(cmapp = 'gnuplot'):
    h_p = smooth(h_prime[indexx_amp],10)
    bet = smooth(beta[indexx_amp],10)
    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)

    x_lims = list([datetime.datetime.strptime(str(amp_b.index[0]), '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime(str(amp_b.index[-1]), '%Y-%m-%d %H:%M:%S')])
    x_lims = mdates.date2num(x_lims)
    
    fig, ax = plt.subplots(figsize = (10, 6))

    cax = ax.imshow(np.log10(elec.T), aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp)
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Start time 24-Jul-2016 11:00 UT')


    cbar = fig.colorbar(cax)
    cbar.set_label('$\mathrm{log_{10}(N_e(m^{-3}))}$')
    fig.autofmt_xdate()
    plt.tight_layout()

#############################
#
# Recombination Coeff
#
############################
def recombo():
    h_p = smooth(h_prime[indexx_amp],10)
    bet = smooth(beta[indexx_amp],10)
    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)
    elec = elec.T


    #file containing info of max peaks in GOES
    a = pd.read_csv('peak_n_e.txt', delimiter = ' ')
    
    #max GOES flux at each peak
    I_max = a['max_I']

    #time between peaks
    dt = 90.

    #elec den at max GOES Peak for each height
    h_elec_max = []
    for i in range(len(elec)):
        ff = elec[i][indd]
        h_elec_max.append(ff)

    alphas = []
    for i in range(len(h_elec_max)):
        alpha = 1./(2*dt*h_elec_max[i])
        alphas.append(alpha)

    alphas = np.array(alphas)
    each_peak = []
    for i in range(len(I_max)):
        each_peak.append(alphas[:,i])

    plotting_lines = False
    if plotting_lines:
	    for i in range(0,len(alphas),40):
		a = make_df(np.array(I_max), np.log10(alphas[i]), name1 = 'Flux Max', name2 = 'Recombination Coeff')
		sns.regplot(x = 'Flux Max', y = 'Recombination Coeff', data = a, scatter_kws={"s": 50}, label = str(h[i])+' km')
		plt.legend()
	    


    a_test = (0.51e-6)*np.exp(-0.165*h)
    for i in range(len(each_peak)):
        plt.plot(h, each_peak[i], label = 'Peak no: '+ str(i))
        plt.plot(h, a_test, ls = '--')
        

    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B
    plot_sur = False
    if plot_sur:


	    AA = []#LISTS OF SLOPES
	    BB = []#LIST OF INTERCEPTS
	    for i in range(0, len(alphas)):
		#plt.plot(I_max, np.log10(alphas[i]), ls = ' ', marker = '.')
		A, B = curve_fit(f, I_max, np.log10(alphas[i]))[0]
		
		AA.append(A)
		BB.append(B)

	    #now fit AA, BB in terms of heights
	    A_slope, A_int = curve_fit(f, h, AA)[0]
	    B_slope, B_int = curve_fit(f, h, BB)[0]

	    def emp_al(h, I_max):
		return (A_slope*h + A_int)*I_max + (B_slope*h+ B_int)

	    II = np.array(I_max)
	    II.sort()
	    test_plot = []
	    for i in range(len(h)):
		test_plot.append(emp_al(h[i], np.array(II)))

	    plt.imshow(test_plot, aspect = 'auto', cmap = 'viridis', extent = [II[0], II[-1], h[-1], h[0]])
	     



def make_df(arr1, arr2, name1 = 'name1', name2 = 'name2'):
	d = {name1 : pd.Series(arr1), name2 : pd.Series(arr2)}
	df = pd.DataFrame(d)
	return df

def plot_corr_e():
    h_p = smooth(h_prime[indexx_amp],10)
    bet = smooth(beta[indexx_amp],10)
    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)
    elec = elec.T

    
    for i in range(0,200,40):
        a = make_df(np.array(gll), np.log10(elec[i][0:-1]), name1 = 'amp', name2 = 'elec')
        sns.regplot(x = 'amp', y = 'elec', data = a, scatter_kws={"s": 80}, label = str(h[i])+' km')
        plt.legend()
    
    	


######################################################
#
# function to plot 3d map of e density of time and h
#
#######################################################
def plot_elec_den_paper(cmapp = 'seismic'):
    h_p = smooth(h_prime[indexx_amp],10)
    bet = smooth(beta[indexx_amp],10)
    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)

    x_lims = list([datetime.datetime.strptime(str(amp_b.index[0]), '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime(str(amp_b.index[-1]), '%Y-%m-%d %H:%M:%S')])
    x_lims = dates.date2num(x_lims)
    
    fig, ax = plt.subplots(2, figsize = (10, 10), sharex = True)
    testy = np.array(amp_b)
    ax[0].plot(sidy.index.to_pydatetime(), sidy, sns.xkcd_rgb["pale red"], label = 'BIRR VLF')
    ax[0].plot(sidy.index, smooth(sidy, 120), color = 'k', lw = 2, label = 'Smoothed BIRR VLF')
    ax[0].set_ylabel('Amplitude in dBV')
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major")
    ax[0].set_title('(a) VLF Amplitude Pulsations')

    ax1 = ax[1].imshow(np.log10(elec.T), aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp)
    ax[1].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major", color = 'grey')
    ax[1].set_ylabel('Altitude (km)')
    ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    ax[1].set_title('(b) Electron Density in Lower Ionosphere')

    cbaxes = fig.add_axes([0.85, 0.08, 0.02, 0.4]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('$\mathrm{log_{10}(N_e(m^{-3}))}$')
    fig.autofmt_xdate()
    plt.tight_layout()




def plot_models():
    fig, ax = plt.subplots(figsize = (10,8))

    h_tick = np.arange(0, 20, 5)
    h_lab = np.arange(60, 80, 5)
    b_tick = np.arange(0, 20, 5)
    b_lab = np.arange(0.3, 0.50, 0.05)

    plt.imshow(amp_l, origin = 'lower', cmap = 'viridis')
    plt.xticks(b_tick, b_lab)
    plt.xlabel('Beta (km$^{-1}$)')
    plt.ylabel('H prime (km)')
    plt.yticks(h_tick, h_lab)
    cbar = plt.colorbar()
    cbar.set_label('Amp (db)')
    plt.title('Modeling Amp with different H and beta')

    fig, ax = plt.subplots(figsize = (10,8))
    plt.imshow(pha_l, origin = 'lower', cmap = 'viridis')
    plt.xticks(b_tick, b_lab)
    plt.xlabel('Beta (km$^{-1}$)')
    plt.ylabel('H prime (km)')
    plt.yticks(h_tick, h_lab)
    cbar = plt.colorbar()
    cbar.set_label('Phase Degrees')
    plt.title('Modeling Phase with different H and beta')
