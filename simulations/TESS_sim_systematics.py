# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:51:06 2017

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

from __future__ import division
import numpy as np
import os

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
mpl.rcParams['font.family'] = 'serif'
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy.interpolate as INT
import pyfits as pf
import random
plt.ioff()
from filters import Epanechnikov_filt
import time
disk_path = '/media/mikkelnl/Elements/'

#===============================================================================
# Code
#===============================================================================

def Momentum_dumps(Time):
    obs_dur = 27.4
    dump_duration = 4.5/(60*24) # days (3.5 minutes)
    dump_period = 3 # days
    no_checks = int(np.ceil(13*obs_dur/dump_period))
    
    dump_idx = np.zeros_like(Time, dtype=bool)    
    for i in range(no_checks):
        instance = i*dump_period
        if instance<Time[0]:
            continue
        elif instance>Time[-1]:
            continue
        else:
            idx = (Time<instance+dump_duration) & (Time>instance)
            
            if not any(idx):
                    idx = np.argmin(np.abs(Time - instance))
                    
            dump_idx[idx] = 1 
        
    return dump_idx

#==============================================================================
# Multiplicative systematics
#==============================================================================

def Sys_multi(ELAT, cad=1800, seed=15):
    
    cad_day = cad/(60*60*24)
    obs_dur = 27.4

    seed_factor = (ELAT<=30)*1e3 + (30<ELAT<=54)*1e4+(54<ELAT<=78)*1e5+(ELAT>78)*1e6 
    Sensitivity_factor = (ELAT<=30)*1 + (30<ELAT<=54)*0.5+(54<ELAT<=78)*0.25+(ELAT>78)*0.1 
    random.seed(a=seed*seed_factor)  
    
    Time_baseline = np.arange(0, 13*obs_dur, cad_day)
    Sys_flux = np.ones_like(Time_baseline)

    for i in range(13):
        idx_time = (Time_baseline>i*obs_dur) & (Time_baseline<=(i+1)*obs_dur)
        Sys_flux[idx_time] = random.normalvariate(1, 0.02*Sensitivity_factor)

    return Time_baseline, Sys_flux
    

def Sys_focus(cad=1800, focus_max_loss=0.01):
    
    cad_day = cad/(60*60*24)
    obs_dur = 27.4    
    Time_baseline = np.arange(0, 13*obs_dur, cad_day)
    
    focus_phase = ((Time_baseline/13.7 + 0.5) % 1) - 0.5
    focus_sigma = 0.1
    focus = 1 - focus_max_loss*np.exp(-0.5*(focus_phase/focus_sigma)**2)
    
    return Time_baseline, focus

#==============================================================================
# Systematic time dependent components
#==============================================================================

def CBV_systematic(ELAT=30, cad=1800, seed=15, path=None):

    cad_day = cad/(60*60*24)
    obs_dur = 27.4

    seed_factor = (ELAT<=30)*1e3 + (30<ELAT<=54)*1e4+(54<ELAT<=78)*1e5+(ELAT>78)*1e6 
    random.seed(a=seed*seed_factor)  
    
    Time_baseline = np.arange(0, 13*obs_dur, cad_day)
    Sys_flux = np.zeros_like(Time_baseline)
    
    if path is None:
        path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/Systematics/CBV'
    files = os.listdir(path)
    
    for i in range(13):
        idx_time = (Time_baseline>i*obs_dur) & (Time_baseline<=(i+1)*obs_dur)
        Sec_time = Time_baseline[idx_time]
        Sec_time -= Sec_time[0]       
        
        cbv_quar = files[random.randint(0, len(files)-1)]
        CBV = pf.open(os.path.join(path, cbv_quar), memmap=True, mode='readonly')
        
        for k in range(10): # Some CBVs aren't good, so we iterate until a good one is found
            try:
                cbv_module = random.randint(1, 84)
                T = np.array(CBV[cbv_module].data['TIME_MJD'])
                T -= T[0] 
                idx = np.isfinite(T)
                T = T[idx]
                if len(T)>len(Sec_time)*cad/(30*60): # CBV are in LC, so if cad is different this must be accounted for
                    break
            except:
                continue
        
        # Make continuous time vector, without the gaps in the CBVs
        dt = np.median(np.diff(T))
        Tnew = np.arange(0, T[-1]+dt, dt)
        
        # Always use first and second CBV, and add third and/or fourth CBV
        cbvs = np.array([1, 2] + [random.randint(3, 4) for j in range(random.randint(1, 2))])
        
        # Make a new gap-filled and summed CBV
        V = np.zeros_like(Tnew)
        q = np.ones_like(Tnew, dtype=bool)
        w = 1
        for j in range(len(cbvs)):
            comp = w*np.array(CBV[cbv_module].data['VECTOR_' + str(cbvs[j])])[idx]
            compn = np.zeros_like(Tnew)
            
            std_V = np.std(np.diff(comp))
            for kk in range(len(compn)):
                idx_close = np.argmin(np.abs(T-Tnew[kk]))
                if np.abs(T[idx_close]-Tnew[kk])<1.5*dt:
                    compn[kk] = comp[idx_close]
                else:
                    compn[kk] = random.normalvariate(comp[idx_close], std_V)
                    # keep a record of wheather or not data existed  
                    q[kk] = 0

            m = Epanechnikov_filt(compn, 15)
            for ii in range(4):
                m = Epanechnikov_filt(compn, 15)
            V += m
            
            # Weight is decreasing for the higher CBVs
            w /= 3
        
  
        # Pick random segment of CBV that matches the lenght of the observing segment  
        LC_len_sec = int(np.ceil(len(Sec_time)*cad/(30*60)))  # CBV are in LC, so if cad is different this must be accounted for with (cad/30)        
        idx_time_start = random.randint(0, len(Tnew)-LC_len_sec-1) 
        T_interp = Tnew[idx_time_start:-1]
        V_interp = V[idx_time_start:-1]
        q_interp = q[idx_time_start:-1]
        
        # Make interpolation of CBV
        Interp = INT.UnivariateSpline(T_interp-T_interp[0], V_interp, k=2, s=0.00001)
        Tgood = T_interp - T_interp[0]

        # Find, in times of observing section, if data existed in original CBV
        idx_good = np.ones_like(Sec_time, dtype=bool) 
        
        print(len(idx_good))
        for jj in range(len(Sec_time)):
            idx_close = np.argmin(np.abs(Tgood-Sec_time[jj]))
            idx_good[jj] = q_interp[idx_close]
        
        Sys_flux_temp = Interp(Sec_time)    
        Sys_flux_temp[~idx_good] = np.nan    
        Sys_flux[idx_time] = Sys_flux_temp 
        print(len(Sys_flux))
    
        
        
#        plt.figure()
#        plt.plot(T, V0, 'k.')
#        plt.plot(Tnew, V, 'b')
#        plt.plot(T_interp, V_interp)
#        plt.plot(Sec_time+Tnew[idx_time_start], Interpolated, 'r')
#        plt.show()
    
    return Time_baseline, Sys_flux

#==============================================================================
# 
#==============================================================================

def CBV_systematic_calculated(ELAT, cad, seed, path=None):
    
    if path is None:
        path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/Systematics/Calculated_sys_curves/'
        
    try:    
        print('Loading systematics')
        # Ecliptic latitude centres of CCDs
        ELATS = np.array([18, 42, 66, 90])
        Elat_idx = np.argmin(np.abs(ELAT-ELATS))
        name = 'Sys_CBV_EL' + str(int(ELATS[Elat_idx])) + 'CAD' + str(int(cad)) + 'SEED' + str(int(seed)) + '.txt'
        T, S = np.loadtxt(os.path.join(path, name), unpack=True)
    except:
        
        print('Calcluating systematics')
        T, S = CBV_systematic(ELAT, cad=cad, seed=seed)
        
        name = 'Sys_CBV_EL' + str(int(ELATS[Elat_idx])) + 'CAD' + str(int(cad)) + 'SEED' + str(int(seed)) + '.txt'
        np.savetxt(os.path.join(path, name), np.column_stack((T, S)))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(T, S, 'k')
        ax.plot(T, S, 'r.')
        ax.set_xlim([0, np.max(T)])
        ax.set_ylim([ -0.05, 0.05])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Sys signal')
        [ax.axvline(x=j*27.4, ls='-', color='b') for j in range(14)]
        name2 = 'Sys_CBV_EL' + str(int(ELATS[Elat_idx])) + 'CAD' + str(int(cad)) + 'SEED' + str(int(seed)) + '.eps'
        fig.savefig(os.path.join(path, name2))
        plt.close('all')
        
        random.seed(a=int(time.time()*1e6)) # Change fixed seed

    return T, S

#==============================================================================
# Noise model as a function of magnitude and position
#==============================================================================

def ZLnoise(gal_lat):
    # RMS noise from Zodiacal background
    rms = (16-10)*(gal_lat/90 -1)**2 + 10 # e-1 / pix in 2sec integration
    return rms
    
def Pixinaperture(Tmag):
    # Approximate relation for pixels in aperture (based on plot in Sullivan et al.)
    pixels = (30 + (((3-30)/(14-7)) * (Tmag-7)))*(Tmag<14) + 3*(Tmag>=14) + np.random.normal(0, 2)
    return np.max([pixels, 3])

def mean_flux_level(Tmag, Teff):    
    # Magnitude system based on Sullivan et al.
    collecting_area = np.pi*(10.5/2)**2 # square cm
    Teff_list = np.array([2450, 3000, 3200, 3400, 3700, 4100, 4500, 5000, 5777, 6500, 7200, 9700]) # Based on Sullivan
    Flux_list = np.array([2.38, 1.43, 1.40, 1.38, 1.39, 1.41, 1.43, 1.45, 1.45, 1.48, 1.48, 1.56])*1e6 # photons per sec; Based on Sullivan
    Magn_list = np.array([306, -191, -202, -201, -174, -132, -101, -80, -69.5, -40, -34.1, 35])*1e-3 #Ic-Tmag (mmag)
    
    
    Flux_int = INT.UnivariateSpline(Teff_list, Flux_list, k=1, s=0)
    Magn_int = INT.UnivariateSpline(Teff_list, Magn_list, k=1, s=0)
    
    Imag = Magn_int(Teff)+Tmag
    Flux = 10**(-0.4*Imag) * Flux_int(Teff) * collecting_area
    
    return Flux

def phot_noise(Tmag, Teff, cad, PARAM, verbose=False):
    
    # Calculate galactic latitude for Zodiacal noise
    
    gc = SkyCoord(lon=PARAM['ELON']*u.degree, lat=PARAM['ELAT']*u.degree, frame='barycentrictrueecliptic')
    gc_gal = gc.transform_to('galactic')
    gal_lat0 = gc_gal.b.deg

    gal_lat = np.arcsin(np.abs(np.sin(gal_lat0*np.pi/180)))*180/np.pi
           
    # Number of 2 sec integrations in cadence
    integrations = cad/2
    
    # Number of pixels in aperture given Tmag
    pixels = int(Pixinaperture(Tmag))
    
    # noise values are in rms, so square-root should be used when factoring up
    Flux_factor = np.sqrt(integrations * pixels) 
        
    # Mean flux level in electrons per cadence    
    mean_level_ppm = mean_flux_level(Tmag, Teff) * cad # electrons
       
    # Shot noise
    shot_noise = 1e6/np.sqrt(mean_level_ppm)
        
    # Read noise
    read_noise = 10 * Flux_factor *1e6/mean_level_ppm # ppm
        
    # Zodiacal noise
    zodiacal_noise = ZLnoise(gal_lat) * Flux_factor *1e6/mean_level_ppm # ppm  
         
    # Systematic noise in ppm
    systematic_noise_ppm = 60 / np.sqrt(cad/(60*60)) # ppm / sqrt(hr)
    
    
    if verbose:
        print('Galactic latitude', gal_lat)
        print('Systematic noise in ppm', systematic_noise_ppm)
        print('Integrations', integrations)  
        print('Pixels', pixels)
        print('Flux factor', Flux_factor)
        print('Mean level ppm', mean_level_ppm)
        print('Shot noise', shot_noise)
        print('Read noise', read_noise)
        print('Zodiacal noise', zodiacal_noise)    

    
    PARAM['Galactic_lat'] = gal_lat
    PARAM['Pixels_in_aper'] = pixels
    
    noise_vals = np.array([shot_noise, zodiacal_noise, read_noise, systematic_noise_ppm])
    return noise_vals, PARAM # ppm per cadence
    


    
    
#==============================================================================
#         
#==============================================================================

if __name__ == "__main__":    
    
    
    
#    T, S= Sys_multi(90, cad=1800, seed=15)
#    T, S = CBV_systematic_calculated(ELAT=40, cad=1800, seed=15)
    T, S = CBV_systematic(ELAT=30, cad=1800, seed=15)
    
    
    
    plt.figure()
    plt.plot(T, S, 'k')
    plt.plot(T, S, 'r.')
    [plt.axvline(x=i*27.4, ls='-', color='b') for i in range(14)]
    plt.show()    
    
    
    #TT = np.linspace(2500, 9700, 100)    
#    print mean_flux_level(0, 9700)    
#        
#    mags = np.linspace(3.5, 16.5, 50)
#    
#    vals = np.zeros([50, 4])
#    
#    for i in range(len(mags)):
#        vals[i,:] = phot_noise(50, 50, mags[i], 5777, 60)    
#        
#    plt.figure()
#    plt.semilogy(mags, vals[:, 0], 'r-')   
#    plt.semilogy(mags, vals[:, 1], 'g--')   
#    plt.semilogy(mags, vals[:, 2], '-')   
#    plt.semilogy(mags, np.sqrt(np.sum(vals**2, axis=1)), 'k-')   
#    plt.axhline(y=60, color='b', ls='--') 
#    
#    plt.axis([3.5, 16.5, 10, 1e5])   
#    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    