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
sns.set_context('paper', font_scale=1.5)
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

msk_amp_flare = msk_amp_real.truncate('2016-07-24 11:00','2016-07-24 16:00')
msk_pha_flare = msk_pha_real.truncate('2016-07-24 11:00','2016-07-24 16:00')


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
sid_flare = sid_dataa.truncate('2016-07-24 11:30','2016-07-24 15:30')

s_a = np.array(sid_flare)
s_a[9598: 9623] = np.nan #for 11:30
#s_a[11398:11423] = np.nan #for 11:00
not_nan = np.logical_not(np.isnan(s_a))
indices = np.arange(len(s_a))
new_sa = np.interp(indices, indices[not_nan], s_a[not_nan])
sidy = Series(new_sa, index = sid_flare.index)

#reading hprime and beta values
res = np.loadtxt('test_h_and_beta.dat')
#res = np.loadtxt('h_and_beta_test_final.dat')
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


#for h_and_beta_test_final.dat
#hh = np.arange(60, 77, 0.1)
#bb = np.arange(0.28, 0.5, 0.01)

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



#plt.plot(amp_b.index, beta[indexx_amp])


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



def plot_dif_h_p():
	h = np.arange(60, 90,1)
	h_prime = [60, 65, 70, 75, 80]
	for i in range(len(h_prime)):
		plt.plot(n_e(h, h_prime[i], 0.3), h, label = 'H\'= '+str(h_prime[i]))
	plt.legend()

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
    x_lims = dates.date2num(x_lims)
    
    fig, ax = plt.subplots(figsize = (10, 6))

    cax = ax.imshow(np.log10(elec.T), aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp)
    ax.xaxis_date()
    date_format = dates.DateFormatter('%H:%M:%S')
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

def fitLine(x, y, alpha=0.05, newx=[]):
    ''' Fit a curve to the data using a least squares 1st order polynomial fit '''
    
    # Summary data
    n = len(x)			   # number of samples     
    
    Sxx = np.sum(x**2) - np.sum(x)**2/n
#    Syy = np.sum(y**2) - np.sum(y)**2/n    # not needed here
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Linefit
    b = Sxy/Sxx
    a = mean_y - b*mean_x
    
    # Residuals
    fit = lambda xx: a + b*xx    
    residuals = y - fit(x)
    
    var_res = np.sum(residuals**2)/(n-2)
    sd_res = np.sqrt(var_res)
    
    # Confidence intervals
    se_b = sd_res/np.sqrt(Sxx)
    se_a = sd_res*np.sqrt(np.sum(x**2)/(n*Sxx))
    
    df = n-2                            # degrees of freedom
    tval = stats.t.isf(alpha/2., df) 	# appropriate t value
    
    ci_a = a + tval*se_a*np.array([-1,1])
    ci_b = b + tval*se_b*np.array([-1,1])

    # create series of new test x-values to predict for
    npts = 100
    px = np.linspace(np.min(x)-0.5,np.max(x)+0.5,num=npts)
    
    se_fit     = lambda x: sd_res * np.sqrt(  1./n + (x-mean_x)**2/Sxx)
    se_predict = lambda x: sd_res * np.sqrt(1+1./n + (x-mean_x)**2/Sxx)
    
    print 'Summary: a={0:5.4f}+/-{1:5.4f}, b={2:5.4f}+/-{3:5.4f}'.format(a,tval*se_a,b,tval*se_b)
    print 'Confidence intervals: ci_a=({0:5.4f} - {1:5.4f}), ci_b=({2:5.4f} - {3:5.4f})'.format(ci_a[0], ci_a[1], ci_b[0], ci_b[1])
    print 'Residuals: variance = {0:5.4f}, standard deviation = {1:5.4f}'.format(var_res, sd_res)
    print 'alpha = {0:.3f}, tval = {1:5.4f}, df={2:d}'.format(alpha, tval, df)
    
    # Return info
    ri = {'residuals': residuals, 
        'var_res': var_res,
        'sd_res': sd_res,
        'alpha': alpha,
        'tval': tval,
        'df': df}
    return px, fit(px), x, y, fit(x), tval*se_fit(px)
from scipy import stats
def paper_recombo():
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
	
    from scipy import stats
    import itertools

    palette = itertools.cycle(sns.color_palette())

    hhh = [0, 51, 101, 151, -1]
    lala = [65, 70, 75, 80, 85]
    for i in range(len(hhh)):
	a = make_df(np.array(I_max*1e6), (alphas[hhh[i]]), name1 = 'Flux Max', name2 = 'Recombination Coeff')
	aa = a.sort(columns = 'Flux Max')
	
	px, fit_px, x, y, fit_x, tval_se_fit_x = fitLine(np.array(aa['Flux Max']), np.array(np.log10(aa['Recombination Coeff'])))
	
        plt.plot(x, 10**(y),color = next(palette), label = str(lala[i])+' km', marker = '.', ls = ' ', ms = 12)
	if i == 0:
	    plt.text(x[0]-0.05*x[0], 10**(y[0])+0.1*10**(y[0]), str(aa.index[0]), fontsize = 15)
	    for j in range(1, len(x)):
	        plt.text(x[j]+0.01*x[j], 10**(y[j])+0.1*10**(y[j]), str(aa.index[j]), fontsize = 15)



	plt.plot(px,10**(fit_px))
	#x.sort()
	#plt.fill_between(x, 10**(fit_x+tval_se_fit_x),10**(fit_x-tval_se_fit_x), alpha = 0.3)
	plt.fill_between(px, 10**(fit_px+tval_se_fit_x),10**(fit_px-tval_se_fit_x), alpha = 0.2)
        #plt.plot(x, 10**(fit_x+tval_se_fit_x))
        #plt.plot(x, 10**(fit_x-tval_se_fit_x))

	plt.xlim(0.6, 7.4)
	plt.xlabel('X-ray Flux in units of $\mathrm{10^{-6}}$ $\mathrm{Wm^{-2}}}$')

	plt.yscale('log')
	plt.legend(loc = 'lower left')
	#plt.xscale('log')
	plt.ylabel('$\\alpha_{eff}$  ($\mathrm{m^{-3}s^{-1}}$)')


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
	    hhh = [0, 51, 101, 151, -1]
	    lala = [65, 70, 75, 80, 85]
	    for i in range(len(hhh)):
		a = make_df(np.array(I_max*1e6), (alphas[hhh[i]]), name1 = 'Flux Max', name2 = 'Recombination Coeff')
		sns.regplot(x = 'Flux Max', y = 'Recombination Coeff', data = a, scatter_kws={"s": 50}, label = str(lala[i])+' km')
		plt.legend(loc = 'lower left')
		plt.xlabel('Flux Peak (x 10$^6$) (Wm$^{-2}$)')
		plt.yscale('log')
		plt.xscale('log')
		plt.ylabel('log10($ \mathrm{\\alpha_{eff} (m^3s^{-1}))}$')
	    

    from scipy import stats
    import itertools
    plotting_l = True
    palette = itertools.cycle(sns.color_palette())
    if plotting_l:
	    hhh = [0, 51, 101, 151, -1]
	    lala = [65, 70, 75, 80, 85]
	    for i in range(len(hhh)):
		a = make_df(np.array(I_max*1e6), (alphas[hhh[i]]), name1 = 'Flux Max', name2 = 'Recombination Coeff')
		aa = a.sort(columns = 'Flux Max')
		slope, intercept, r_value, p_value, std_err = stats.linregress(aa['Flux Max'], np.log10(aa['Recombination Coeff']))
		plt.plot(np.array(aa['Flux Max']), np.array(aa['Recombination Coeff']), marker = '.', ls = ' ', ms = 12, color = next(palette), label = str(lala[i])+' km')
		plt.plot(np.array(aa['Flux Max']), np.array(10**(slope*aa['Flux Max'] + intercept)))
		#plt.legend(loc = 'lower left')
		plt.xlabel('X-ray Flux in units of $\mathrm{10^{-6} Wm^{-2}}$')
		plt.yscale('log')
		plt.legend(loc = 'lower left')
		#plt.xscale('log')
		plt.ylabel('$ \\alpha_{eff} \mathrm{(m^3s^{-1})}$')


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
    


max_e = [4.2e8, 1.6e9, 3.38e9, 1.2e9]
min_e = [2e8, 3.4e8, 1.4e9, 1.2e9]
max_g = [1.6e-6,2.5e-6, 1.5e-6,  5.6e-6]
min_g = [ 4.2e-7, 1.5e-6, 1.6e-6,1.86e-6]   	



def elec_height():

    h_p = smooth(h_prime[indexx_amp],10)
    bet = smooth(beta[indexx_amp],10)

    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)
    elec = elec.T

    fig, ax = plt.subplots()
    ax.plot(amp_b.index, elec[0], label = '65km')
    ax.plot(amp_b.index, elec[51], label = '70km')
    ax.plot(amp_b.index, elec[101], label = '75km')
    ax.plot(amp_b.index, elec[151], label = '80km')
    ax.plot(amp_b.index, elec[199], label = '85km')
    ax.xaxis_date()
    date_format = dates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)

    ax.grid()

######################################################
#
# function to plot 3d map of e density of time and h
#
#######################################################
def plot_elec_den_paper(cmapp = 'magma'):
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
    
    fig, ax = plt.subplots(2, figsize = (7, 10), sharex = True)
    testy = np.array(amp_b)
    ax[0].plot(sidy.index.to_pydatetime(), sidy, sns.xkcd_rgb["pale red"], label = 'BIRR VLF')
    ax[0].plot(sidy.index, smooth(sidy, 120), color = 'k', lw = 2, label = 'Smoothed BIRR VLF')
    ax[0].set_ylabel('Amplitude (dB)')
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major")
    ax[0].set_title('VLF Amplitude Pulsations')

    ax1 = ax[1].imshow(np.log10(elec.T), aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp)
    ax[1].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major", color = 'grey')
    ax[1].set_ylabel('Altitude (km)')
    ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    ax[1].set_title('Electron Density in Lower Ionosphere')

    cbaxes = fig.add_axes([0.85, 0.08, 0.02, 0.4]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('$\mathrm{log_{10}(N_e(m^{-3}))}$')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(right = 0.82)

import matplotlib.colors as colors   
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_elec_den_paper2(cmapp = 'magma'):
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
    
    fig, ax = plt.subplots(3, figsize = (6, 16), sharex = True)
    testy = np.array(amp_b)
    ax[0].plot(sidy.index.to_pydatetime(), sidy, sns.xkcd_rgb["pale red"], label = 'BIRR VLF')
    ax[0].plot(sidy.index, smooth(sidy, 120), color = 'k', lw = 2, label = 'Smoothed BIRR VLF')
    ax[0].set_ylabel('Amplitude (dB)')
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major")
    #ax[0].set_title('VLF Amplitude Pulsations')
    ax[0].text('2016-07-24 11:41', 62.8, 'a', weight = 'bold', fontsize = 18)
    

    ax1 = ax[1].imshow(elec.T, aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp, norm = colors.LogNorm())
    ax[1].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major", color = 'grey')
    ax[1].set_ylabel('Altitude (km)')
    ax[1].text('2016-07-24 11:41', 83, 'b', weight = 'bold', color = 'white', fontsize = 18)
    #ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    #ax[1].set_title('Electron Density in Lower Ionosphere')
    #divider = make_axes_locatable(ax[1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbaxes = fig.add_axes([0.82, 0.375, 0.023, 0.299]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('Electron Density (m$^{-3}$)')
   
    ax[2].plot(amp_b.index, elec[:,199], label = '85 km')
    #ax[2].plot(amp_b.index, elec[:,151], label = '80 km')
    ax[2].plot(amp_b.index, elec[:,101], label = '75 km')
    #ax[2].plot(amp_b.index, elec[:,51], label = '70 km')
    ax[2].plot(amp_b.index, elec[:,0], label = '65 km')

    
    ax[2].xaxis.set_major_formatter(date_format)
    ax[2].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[2].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[2].xaxis.grid(True, which="major")
    ax[2].set_xlabel('Start time 24-Jul-2016 11:30 UT')

    ax[2].set_xlim(amp_b.index[0], amp_b.index[-1])
    ax[2].text('2016-07-24 11:41', 5e10, 'c', weight = 'bold', fontsize = 18)
    ax[2].set_yscale('log')
    ax[2].set_ylabel('Electon Density (m$^{-3}$)')
    ax[2].legend(loc = 'upper right')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(right = 0.82)
   
def plot_h_b_rev():
    h_p = Series(smooth(h_prime[indexx_amp],10), index = amp_b.index)
    bet = Series(smooth(beta[indexx_amp],10), index = amp_b.index)
    fig, ax = plt.subplots(2, figsize = (7, 10),sharex = True)
    ax[0].plot(h_p, label = 'H\'', color = 'k')
    ax[1].plot(bet, label = r'$\beta$', color = 'grey', lw = 1.5)
    ax[1].set_ylabel(r'$\beta$ (km$^{-1}$)')
    ax[0].set_ylabel('H\' (km)')

    date_format = dates.DateFormatter('%H:%M')
    ax[0].xaxis.set_major_formatter(date_format)
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].set_xlim(amp_b.index[0], amp_b.index[-1])
    ax[0].xaxis.grid(True, which="major", color = 'grey', lw = 0.2)
    ax[1].xaxis.grid(True, which="major", color = 'grey', lw = 0.2)
    ax[0].legend()
    ax[1].legend(loc = 'upper left')
    ax[1].set_xlabel('Start time 24-Jul-2016 11:30 UT')

def plot_elec_den_paper_rev(cmapp = 'magma'):
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
    
    fig, ax = plt.subplots(2, figsize = (7, 10), sharex = True)
    testy = np.array(amp_b)

   
    ax1 = ax[0].imshow(elec.T, aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp, norm = colors.LogNorm())
    ax[0].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[0].xaxis.set_major_formatter(date_format)
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major", color = 'grey')
    ax[0].set_ylabel('Altitude (km)')
    ax[0].text('2016-07-24 11:41', 83, 'a', weight = 'bold', color = 'white', fontsize = 22)
    #ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    #ax[1].set_title('Electron Density in Lower Ionosphere')
    #divider = make_axes_locatable(ax[0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbaxes = fig.add_axes([0.82, 0.54, 0.03, 0.44]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('Electron Density (m$^{-3}$)')
   
    ax[1].plot(amp_b.index, elec[:,199], label = '85 km')
    #ax[2].plot(amp_b.index, elec[:,151], label = '80 km')
    ax[1].plot(amp_b.index, elec[:,101], label = '75 km')
    #ax[2].plot(amp_b.index, elec[:,51], label = '70 km')
    ax[1].plot(amp_b.index, elec[:,0], label = '65 km')

    
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major")
    ax[1].set_xlabel('Start time 24-Jul-2016 11:30 UT')

    ax[1].set_xlim(amp_b.index[0], amp_b.index[-1])
    ax[1].text('2016-07-24 11:41', 1.5e11, 'b', weight = 'bold', fontsize = 22)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Electon Density (m$^{-3}$)')
    ax[1].legend(loc = 'upper right', fontsize = 15)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(right = 0.81, left = 0.14, top = 0.98, bottom = 0.06)


def plot_elec_den_paper_review_h_p2(cmapp = 'magma'):
    h_p = smooth(h_prime[indexx_amp],10)-0.51
    bet = smooth(beta[indexx_amp],10)
    
    h  = np.arange(65, 85,0.1)
    elec = []
    for i in range(len(h_p)):
        test = n_e(h, h_p[i], bet[i])
        elec.append(test)

    elec = np.array(elec)

    x_lims = list([datetime.datetime.strptime(str(amp_b.index[0]), '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime(str(amp_b.index[-1]), '%Y-%m-%d %H:%M:%S')])
    x_lims = dates.date2num(x_lims)
    
    fig, ax = plt.subplots(3, figsize = (6, 16), sharex = True)
    testy = np.array(amp_b)
    ln1 = ax[0].plot(amp_b.index, h_p, label = 'H\'', color = 'k')
    axy2 = ax[0].twinx()
    ln2 = axy2.plot(amp_b.index, bet, label = r'$\beta$', color = 'grey')
    ax[0].set_ylabel('H\' (km)')
    axy2.set_ylabel(r'$\beta$ (km$^{-1})$', color = 'grey')
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major")
    #ax[0].set_title('VLF Amplitude Pulsations')
    ax[0].text('2016-07-24 11:41', 73, 'a', weight = 'bold', fontsize = 18)
    ax[0].set_ylim(62, 74.1)

    ax[0].text('2016-07-24 15:05', 71, 'H\'')
    axy2.text('2016-07-24 15:05', 0.325, r'$\beta$', color = 'grey')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    #ax[0].legend(lns, labs, loc = 'center right')
    ax1 = ax[1].imshow(elec.T, aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp, norm = colors.LogNorm())
    ax[1].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major", color = 'grey')
    ax[1].set_ylabel('Altitude (km)')
    ax[1].text('2016-07-24 11:41', 83, 'b', weight = 'bold', color = 'white', fontsize = 18)
    #ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    #ax[1].set_title('Electron Density in Lower Ionosphere')
    #divider = make_axes_locatable(ax[1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbaxes = fig.add_axes([0.82, 0.375, 0.023, 0.299]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('Electron Density (m$^{-3}$)')
   
    ax[2].plot(amp_b.index, elec[:,199], label = '85 km')
    #ax[2].plot(amp_b.index, elec[:,151], label = '80 km')
    ax[2].plot(amp_b.index, elec[:,101], label = '75 km')
    #ax[2].plot(amp_b.index, elec[:,51], label = '70 km')
    ax[2].plot(amp_b.index, elec[:,0], label = '65 km')

    
    ax[2].xaxis.set_major_formatter(date_format)
    ax[2].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[2].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[2].xaxis.grid(True, which="major")
    ax[2].set_xlabel('Start time 24-Jul-2016 11:30 UT')

    ax[2].set_xlim(amp_b.index[0], amp_b.index[-1])
    ax[2].text('2016-07-24 11:41', 1.5e11, 'c', weight = 'bold', fontsize = 18)
    ax[2].set_yscale('log')
    ax[2].set_ylabel('Electon Density (m$^{-3}$)')
    ax[2].legend(loc = 'upper right')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(right = 0.82)
   


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



def thesis_hb():
    h_p = Series(smooth(h_prime[indexx_amp],10), index = amp_b.index)
    bet = Series(smooth(beta[indexx_amp],10), index = amp_b.index)
    fig, ax = plt.subplots(2, figsize = (6, 10),sharex = True)
    ax[0].plot(h_p, label = '$H$\'', color = 'k')
    ax[1].plot(bet, label = r'$\beta$', color = 'grey', lw = 1.5)
    ax[1].set_ylabel(r'$\beta$ (km$^{-1}$)')
    ax[0].set_ylabel('$H$\' (km)')

    date_format = dates.DateFormatter('%H:%M')
    ax[0].xaxis.set_major_formatter(date_format)
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].set_xlim(amp_b.index[0], amp_b.index[-1])
    ax[0].xaxis.grid(True, which="major", color = 'grey', lw = 0.2)
    ax[1].xaxis.grid(True, which="major", color = 'grey', lw = 0.2)
    ax[0].legend(loc = 'upper right')
    ax[1].legend(loc = 'upper right')
    ax[1].set_xlabel('Start time 24-Jul-2016 11:30 UT')

    ax[1].text(0.03, 0.94, 'b.',

        transform=ax[1].transAxes)
    ax[0].text(0.03, 0.94, 'a.',

        transform=ax[0].transAxes)


    #ax[1].text('2016-07-24 15:05', 0.325, r'$\beta$', color = 'grey')
    #ax[0].text('2016-07-24 15:05', 71, 'H\'')
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.01)










def thesis_elec(cmapp = 'magma_r'):
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
    
    fig, ax = plt.subplots(2, figsize = (7, 10), sharex = True)
    testy = np.array(amp_b)

   
    ax1 = ax[0].imshow(elec.T, aspect = 'auto', origin = 'lower', extent = [x_lims[0], x_lims[1], 65, 85], cmap = cmapp, norm = colors.LogNorm())
    ax[0].xaxis_date()
    date_format = dates.DateFormatter('%H:%M')
    ax[0].xaxis.set_major_formatter(date_format)
    ax[0].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[0].xaxis.grid(True, which="major", color = 'grey')
    ax[0].set_ylabel('Altitude (km)')
    #ax[0].text('2016-07-24 11:41', 83, 'a')#, weight = 'bold', color = 'white', fontsize = 22)
    #ax[1].set_xlabel('Start time 24-Jul-2016 11:00 UT')
    #ax[1].set_title('Electron Density in Lower Ionosphere')
    #divider = make_axes_locatable(ax[0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    cbaxes = fig.add_axes([0.82, 0.535, 0.03, 0.445]) 
    cb = plt.colorbar(ax1, cax = cbaxes)  

    #cbar = fig.colorbar(cax, orientation = 'horizontal', position = 'top')
    cb.set_label('Electron Density (m$^{-3}$)')
   
    ax[1].plot(amp_b.index, elec[:,199], label = '85 km')
    #ax[2].plot(amp_b.index, elec[:,151], label = '80 km')
    ax[1].plot(amp_b.index, elec[:,101], label = '75 km')
    #ax[2].plot(amp_b.index, elec[:,51], label = '70 km')
    ax[1].plot(amp_b.index, elec[:,0], label = '65 km')

    
    ax[1].xaxis.set_major_formatter(date_format)
    ax[1].xaxis.set_major_locator(dates.MinuteLocator(interval =30))
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%H.%M'))
    ax[1].xaxis.grid(True, which="major")
    ax[1].set_xlabel('Start time 24-Jul-2016 11:30 UT')

    ax[1].set_xlim(amp_b.index[0], amp_b.index[-1])
    #ax[1].text('2016-07-24 11:41', 1.5e11, 'b.')#, fontsize = 22)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Electon Density (m$^{-3}$)')
    ax[1].legend(loc = 'upper right', fontsize = 15)



    ax[1].text(0.03, 0.94, 'b.',

        transform=ax[1].transAxes)
    ax[0].text(0.03, 0.94, 'a.',

        transform=ax[0].transAxes)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.subplots_adjust(right = 0.81, left = 0.14, top = 0.98, bottom = 0.06)





