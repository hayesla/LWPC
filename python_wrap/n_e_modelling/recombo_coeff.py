import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks',{'xtick.direction':'in','ytick.direction':'in'})
sns.set_context('paper', font_scale = 1.5)


#file with peak number, electron density at I_max and I_max
a = pd.read_csv('peak_n_e.txt', delimiter = ' ')

n_e = a['electron_den']
I_max = a['max_I']

dt = 90. #seconds between peak in X-ray and SID

alpha = 1./(2*dt*n_e)

plt.plot(I_max, alpha, ls = ' ', marker = '.', ms = 15)
plt.xlabel('Peak Flux of Pulsations (Wm$\mathrm{^{-2}}$)')
plt.ylabel('Recombination Coefficient $\\alpha$ ($\mathrm{m^3s^{-1}}$)')
plt.yscale('log')
plt.xscale('log')
plt.title('Effective Recombination Coefficient')

savey = False
if savey:
	plt.savefig('electron_recombo.png')

def make_df(arr1, arr2, name1 = 'name1', name2 = 'name2'):
	d = {name1 : pd.Series(arr1), name2 : pd.Series(arr2)}
	df = pd.DataFrame(d)
	return df


d = {'i_max' : pd.Series(I_max), 'alpha': pd.Series(np.log10(alpha))}
df = pd.DataFrame(d)

sns.regplot(x = 'i_max', y = 'alpha', data = df)

from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B
plt.clf()
A,B = curve_fit(f, I_max, np.log10(alpha))[0]
plt.plot(I_max, np.log10(alpha), ls = ' ', marker = '.', ms = 15)
plt.plot(I_max, A*I_max + B)
