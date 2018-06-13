# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:01:17 2017

@author: Dr. Mikkel N. Lund
"""
#===============================================================================
# Packages
#===============================================================================

from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
import random
import time
mpl.rcParams['font.family'] = 'serif'


from TESS_sim_coverage import sky_coverage_calculated
from TESS_sim_systematics import CBV_systematic_calculated, Sys_multi, Sys_focus, Momentum_dumps
plt.ioff()

disk_path = '/media/mikkelnl/Elements/'
#disk_path = '/media/Elements/'
#===============================================================================
# Code
#===============================================================================

def time_vector(cad=1800, ELAT=None, verbose=False, gaps=None, seed=15, PARAM=None):
    
    if PARAM is None:
        PARAM = {}
        PARAM['ELAT'] = ELAT
        
    cad_day = cad/(60*60*24)
    obs_dur = 27.4
    Time_baseline = np.arange(0, 13*obs_dur, cad_day)
    Quality_baseline = np.zeros_like(Time_baseline, dtype=int)
    
    if gaps is None:
        gaps_earth = []   
        gaps_safe = []
        
        # Use fixed seed to allow multiple stars to have same time 
        random.seed(a=seed)  
        PARAM['Tvec_seed'] = seed
        
        # Manually defined Safe mode events    
        gaps_safe.append((100-5, 100+5))
        gaps_safe.append((220-5, 220+5))
    
        # Define gaps from downlinks every 13.7 days 
        # Each gap has a duration between 4-16 hours     
        min_gap_duration = 4/24
        max_gap_duration = 16/24
        
        
        for i in range(13):
            r1=random.random(); g1 = (r1<=0.6)*min_gap_duration + (r1>0.6)*random.uniform(min_gap_duration, max_gap_duration)
            r2=random.random(); g2 = (r2<=0.6)*min_gap_duration + (r2>0.6)*random.uniform(min_gap_duration, max_gap_duration)
            r3=random.random(); g3 = (r3<=0.6)*min_gap_duration + (r3>0.6)*random.uniform(min_gap_duration, max_gap_duration)
            if i==0:
                gaps_earth.append(((i+1)*obs_dur/2 - g1/2, (i+1)*obs_dur/2 + g1/2))
                gaps_earth.append(((i+1)*obs_dur - g2/2, (i+1)*obs_dur))
            else:
                gaps_earth.append(((i+3)*obs_dur/2 - g1/2, (i+3)*obs_dur/2 + g1/2))
                gaps_earth.append((i*obs_dur, i*obs_dur + g2/2))
                gaps_earth.append(((i+1)*obs_dur - g3/2, (i+1)*obs_dur))
      

    random.seed(a=int(time.time()*1e6)) # Change fixed seed from time vector generation

    # Identify gap indices         
    idx_e = np.zeros_like(Time_baseline, dtype=bool)
    for Tg in gaps_earth:
        idx_e += ((Time_baseline>Tg[0]) & (Time_baseline<Tg[1]))
    idx_s = np.zeros_like(Time_baseline, dtype=bool)
    for Tg in gaps_safe:
        idx_s += ((Time_baseline>Tg[0]) & (Time_baseline<Tg[1]))    
        
    
    # Mark positions of gaps in quality vector
    Quality_baseline[idx_e] += 8 # Spacecraft is in Earth point midway in observing sector
    Quality_baseline[idx_s] += 2 # Spacecraft experiences safe mode


    # Split time in observing sectors
    Time_sectors = np.array([Time_baseline[(Time_baseline>=i*obs_dur) & (Time_baseline<(i+1)*obs_dur)] for i in range(13)])
    Quality_sectors = np.array([Quality_baseline[(Time_baseline>=i*obs_dur) & (Time_baseline<(i+1)*obs_dur)] for i in range(13)])
    
    # Number of sectors in which star is observed
    D, PARAM = sky_coverage_calculated(PARAM, verbose=verbose)
    
    no_sectors = int(np.ceil(D/obs_dur))
    
    # Assign sectors
    idx = random.randint(0, 13-no_sectors)


    Time_obs = np.concatenate(Time_sectors[idx:idx+no_sectors])
    
    
    ELON = random.normalvariate(np.mean(Time_obs), (np.max(Time_obs)-np.min(Time_obs))*0.5)*360/365
    if ELON>np.max(Time_obs)*360/365:
        ELON = np.max(Time_obs)*360/365
    if ELON<np.min(Time_obs)*360/365:
        ELON = np.min(Time_obs)*360/365        

    PARAM['ELON'] = ELON
    
    
    Quality_obs = np.concatenate(Quality_sectors[idx:idx+no_sectors])
    
    # Momentum dumps every 3 days
    dump_idx = Momentum_dumps(Time_obs)
    Quality_obs[dump_idx] += 32+4 # Desaturation and Coarse point
    
    # Load additive systematics and updata Quality vector
    Tsys, Sysa = CBV_systematic_calculated(ELAT, cad, seed)
    
    idx_sys = (Tsys>=Time_obs[0]) & (Tsys<=Time_obs[-1])
    Sysa = Sysa[idx_sys]
    
    Sys_gap = np.isnan(Sysa)
    Quality_obs[Sys_gap] += 256 # Manual exclude
    
    # Load multiplicative and focus systematics
    _, Sysm = Sys_multi(PARAM['ELAT'], cad, seed)
    _, Sysf = Sys_focus(cad)
    Sysm = Sysm[idx_sys]
    Sysf = Sysf[idx_sys]

    return Time_obs, Quality_obs, PARAM, Sysa, Sysm, Sysf


    
#==============================================================================
#         
#==============================================================================

if __name__ == "__main__":
    
#    calc_coverage()        
    
#    D = sky_coverage_calculated(RA=20, DEC=90, ELAT=50, verbose=True)
   
    T, Q, PARAM, Sa, Sm, Sf = time_vector(cad=1800, ELAT=80, verbose=False, seed=15)
    
    
    plt.figure()
    plt.plot(T, Sa)
    plt.plot(T, Sm)
    plt.plot(T, Sf)
    
    
    plt.show()
    print(T, len(T))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    