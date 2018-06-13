#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:51:38 2011

@author: mikkelnl
"""

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import scipy.optimize as op
import itertools
import os
import numpy as np
from numpy import median
from pylab import *
import sys
import scipy.signal as sig




def Run_mean(window_len,Power):
    #Running mean:
    Power = np.array(Power)
    power_extended=np.r_[Power[(window_len/2)-1:0:-1],Power,Power[-1:-(window_len/2)-1:-1]]
    Power_smo=[0.0]*(len(power_extended)-(window_len-1))
    
    for i in range(len(Power_smo)):
        Power_smo[i]=np.mean(np.array(power_extended[i:i+(window_len-1)]))  
    return Power_smo


def MAD(data):
    Med = median(data)
    return median(abs(data-Med))
    
    
def Run_median(window_len,Power1, percent=50):
    
    if np.mod(window_len/2,2)!=0:
        win_ext=np.ceil(window_len/2)+1
    else:
        win_ext=window_len/2
        
    Power = np.array(Power1)
    

    power_extended=np.r_[Power[int(win_ext):0:-1],Power,Power[-1:int(-win_ext-1):-1]]
    Power_smo=[0.0]*len(Power)
    
    for i in range(len(Power_smo)):
        #Power_smo[i]=np.median(np.array(power_extended[i:i+int(2*win_ext)]))  
	Power_smo[i]=np.percentile(np.array(power_extended[i:i+int(2*win_ext)]), percent)  
    return np.array(Power_smo)  

#    power_extended=np.r_[Power[(window_len/2)-1:0:-1],Power,Power[-1:-(window_len/2)-1:-1]]
#    Power_smo=[0.0]*(len(power_extended)-(window_len-1))
#    
#    for i in range(len(Power_smo)):
#        Power_smo[i]=np.median(np.array(power_extended[i:i+(window_len-1)]))  
#    return Power_smo    

def Epanechnikov_filt(Signal, win):
    
    """
    Signal: Input signal to be filtered \n
    win: window width in number of points
    """
    
    """
    ‘full’: By default, mode is ‘full’. This returns the convolution at each point of overlap, 
    with an output shape of (N+M-1,). At the end-points of the convolution, the signals do 
    not overlap completely, and boundary effects may be seen.
    ‘same’: Mode same returns output of length max(M, N). Boundary effects are still visible.
    ‘valid’: Mode valid returns output of length max(M, N) - min(M, N) + 1. 
    The convolution product is only given for points where the signals overlap completely. 
    Values outside the signal boundary have no effect.
    """
    
    def Epanechnikov(win):
        u=(np.array(range(win))-np.array(range(win)).max()/2) / (np.array(range(win)).max() / 2)
        return (3/4)*(1-u**2)

    w = Epanechnikov(win)
    Sig = np.r_[Signal[int(win)-1:0:-1],Signal,Signal[-1:-int(win):-1]]
    y = np.convolve(w/w.sum(),Sig, mode='valid')
    
    y = y[int(np.floor(win/2)):len(y)-int(np.floor(win/2))] 
 
    return y 
    
def Tricube_filt(Signal, win):
    
    """
    Signal: Input signal to be filtered \n
    win: window width in number of points
    """
    
    """
    ‘full’: By default, mode is ‘full’. This returns the convolution at each point of overlap, 
    with an output shape of (N+M-1,). At the end-points of the convolution, the signals do 
    not overlap completely, and boundary effects may be seen.
    ‘same’: Mode same returns output of length max(M, N). Boundary effects are still visible.
    ‘valid’: Mode valid returns output of length max(M, N) - min(M, N) + 1. 
    The convolution product is only given for points where the signals overlap completely. 
    Values outside the signal boundary have no effect.
    """
    
    def Tricube(win):
        u=(np.array(range(win))-np.array(range(win)).max()/2) / (np.array(range(win)).max() / 2)
        return (70/81)*(1-np.abs(u)**3)**3

    w = Tricube(win)
    Sig = np.r_[Signal[int(win)-1:0:-1],Signal,Signal[-1:-int(win):-1]]
    y = np.convolve(w/w.sum(),Sig, mode='valid')
    
    y = y[int(np.floor(win/2)):len(y)-int(np.floor(win/2))] 
 
    return y     
 
    
def savitzky_golay(y, window_size, order, deriv=0):
    r'''

    from: http://www.scipy.org/Cookbook/SavitzkyGolay
    
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
	
    For example :math:`\sigma`. Or:
    
    .. math::

	\Delta =\frac{\sigma}{\sum_{i=1}^N i}

    ----------
    y : array_like, shape (N,)
         the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
    Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
         the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
         W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
         Cambridge University Press ISBN-13: 9780521880688

    '''
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
        
        
    if window_size % 2 != 1 or window_size < 1:
        window_size += 1
#        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')
    


def SmoothFilter(Fre, Sig_start1, win, Filter='gaussian', param=3, f=1):
    
    
    '''
    Possible filters:
        
    boxcar, triang, blackman, hamming, hanning, 
    bartlett, parzen, bohman, blackmanharris, 
    nuttall, barthann, kaiser (needs beta), 
    gaussian (needs std), general_gaussian (needs power, width), 
    slepian (needs width), chebwin (needs attenuation)
    '''
    #print win
    if f==1:
        step=Fre[10]-Fre[9]
        win = int(round(win/step))
        param = int(round(param/step))
    
    #print win
    if np.mod(win/2,2)!=0:
        win_ext=np.ceil(win/2)+1
    else:
        win_ext=win/2
        
    Sig_start = np.array(Sig_start1)
    #print len(Sig_start)
    Sig=np.r_[ Sig_start[int(win_ext):0:-1] , Sig_start , Sig_start[-1:int(-win_ext-1):-1] ]   
    
    Sig = np.array(Sig)

    if Filter == 'gaussian' or Filter == 'kaiser' or Filter == 'chebwin' or Filter == 'slepian':
        Gauss = sig.get_window((Filter,param), win)
    else:
        Gauss = sig.get_window(Filter, win)
    
    #print len(sig.convolve(Sig, Gauss, mode='same'))
    Sig_smo = np.array([x/win for x in sig.convolve(Sig, Gauss, mode='same')])
    #print len(Sig_smo)
    Len = len(Sig_start[int(win_ext):0:-1]) 
    Sig_smo =list(Sig_smo)
    del Sig_smo[0:Len]
    del Sig_smo[-Len::]

    
    Sig_smo=np.array(Sig_smo)

    return Sig_smo    
    

def banana_filter(t, x, win):
    
#    if np.mod(win/2,2)!=0:
#        win_ext=np.ceil(win/2)+1
#    else:
#        win_ext=win/2
#    
#    Sig_start = np.array(x)
#    Sig=np.r_[ Sig_start[int(win_ext):0:-1] , Sig_start , Sig_start[-2:int(-win_ext-2):-1] ]
#    Sig = np.array(Sig)
#    
#    Gauss = sig.get_window('boxcar', win)   
#    Sig_smo = np.array([x/win for x in sig.convolve(Sig, Gauss, mode='same')])
    
    x = np.array(x)
    N = len(t)
    tmin = np.min(t)
    dt = (np.max(t)-tmin)/N 
    win2 = 2*win
    winH = win/2
    D = np.zeros(N) 
    t2 = np.zeros(N)
    
    
    for i in xrange(N):
        if np.mod(i,10000)==0:
            print i
        
        ti = i*dt+tmin
        m1 = np.ma.masked_inside(t, ti-winH-win2, ti-winH, copy=True)

        if np.any(m1.mask):
            W1 = np.mean( x[m1.mask] )
        else:
            W1 = 0
        
        m2 = np.ma.masked_inside(t, ti+winH, ti+winH+win2, copy=True) 
        if np.any(m2.mask):
            W2 = np.mean( x[m2.mask] )
        else:
            W2 = 0
        
        m3 = np.ma.masked_inside(t, ti-winH, ti+winH, copy=True) 
        if np.any(m3.mask):
            C = np.mean( x[m3.mask] )
        else:
            C = 0
        
        D[i] = C - (W1 + W2)/2
        t2[i] = ti
    
    return t2, D




  
