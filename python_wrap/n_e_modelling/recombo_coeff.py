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


d = {'i_max' : pd.Series(I_max), 'alpha': pd.Series((alpha))}
df = pd.DataFrame(d)
test = df.sort(columns = 'i_max')


sns.regplot(x = 'i_max', y = 'alpha', data = df)


from scipy import stats
x = test['i_max']
y = np.log10(test['alpha'])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plt.plot(test['i_max'], test['alpha'])
plt.plot(test['i_max'], 10**(slope*test['i_max'] + intercept))


from scipy.optimize import curve_fit

def fitLine(x, y, alpha=0.05, newx=[], plotFlag=1):
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
    px = np.linspace(np.min(x),np.max(x),num=npts)
    
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
    
    if plotFlag == 1:
        # Plot the data
        plt.figure()
        
        plt.plot(px,10**(fit(px)),'k', label='Regression line')
        plt.plot(x, 10**(y),'r.', label='Sample observations')
        
        x.sort()
        limit = (1-alpha)*100
        plt.plot(x, 10**(fit(x)+tval*se_fit(x)), 'r--', label='Confidence limit ({0:.1f}%)'.format(limit))
        plt.plot(x, 10**(fit(x)-tval*se_fit(x)), 'r--')
        
        plt.plot(x, 10**(fit(x)+tval*se_predict(x)), 'c--', label='Prediction limit ({0:.1f}%)'.format(limit))
        plt.plot(x, 10**(fit(x)-tval*se_predict(x)), 'c--')

        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Linear regression and confidence limits')
        
        # configure legend
        plt.legend(loc=0)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=10)

        # show the plot
        plt.show()
        


def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B
plt.clf()
A,B = curve_fit(f, I_max, np.log10(alpha))[0]
#plt.plot(I_max, (alpha), ls = ' ', marker = '.', ms = 15)
#plt.plot(I_max, 10**(A*I_max + B))
