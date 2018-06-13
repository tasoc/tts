# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:09:47 2018

@author: Dr. Mikkel N. Lund
"""

#===============================================================================
# Packages
#===============================================================================

from __future__ import division
import numpy as np
import os, sys
import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
import random

from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
mpl.rcParams['font.family'] = 'serif'
import glob
import time

from TESS_sim_tvec import time_vector
from TESS_sim_systematics import mean_flux_level, phot_noise

import json
import copy, shutil, zipfile
from bisect import bisect

Vizier.ROW_LIMIT = -1
Simbad.ROW_LIMIT = -1

plt.ioff()

disk_path = '/media/mikkelnl/Elements/'

#===============================================================================
# Code
#===============================================================================

RSun = 6.9598e8 # m
MSun_kg = 1.989e30 # kg
Grav = 6.67408e-11 # m3 kg-1 s-2

flux_header = \
"""
Column 1: Time (days)
Column 2: Flux (counts)
Column 3: Quality flag
"""

#==============================================================================
# 
#==============================================================================

def save_LC_data(DATA, PARAM):
    
    if not os.path.exists(PARAM['LC_save_path']):
        os.makedirs(PARAM['LC_save_path'])
            
            
    with open(os.path.join(PARAM['LC_save_path'], PARAM['ID'] + '.json'), 'w') as fp:
        json.dump(PARAM, fp, sort_keys=True, indent=4)
        
    if PARAM['Save_LC']:        
        D = np.column_stack((DATA[:,0], DATA[:,3], DATA[:,4]))        
        np.savetxt(os.path.join(PARAM['LC_save_path'], PARAM['ID'] + '.noisy'), D, fmt='%5.9f %5.9f %i', header=flux_header)
        
        D2 = np.column_stack((DATA[:,0], DATA[:,1], DATA[:,4]))        
        np.savetxt(os.path.join(PARAM['LC_save_path'], PARAM['ID'] + '.clean'), D2, fmt='%5.9f %5.9f %i', header=flux_header)
    
#==============================================================================
# 
#==============================================================================

def plot_LC_data(T, N, C, NS, PARAM):
  
    figname = PARAM['ID'] + '; ' + PARAM['Sim_Category'] + ' LC'
    fig = plt.figure(figname, figsize=(15,8))
    fig.subplots_adjust(top=0.95, left=0.07, bottom=0.1, right=0.98, wspace=0.23)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    [ax1.axvline(x=T[i], color='0.5') for i in range(len(T)) if np.isnan(NS[i])]
    [ax2.axvline(x=T[i], color='0.5') for i in range(len(T)) if np.isnan(NS[i])]
    ax2.plot(T, (NS/np.nanmedian(NS) -1)*1e6, 'k')
    ax1.plot(T, (N/np.nanmedian(N) -1)*1e6, 'k')
    ax1.plot(T, (C/np.nanmedian(C) -1)*1e6, 'r')
    ax1.set_xlim([T[0], T[-1]])
    ax2.set_xlim([T[0], T[-1]])   
    ax1.set_xlabel('Time (days)')
    ax2.set_xlabel('Time (days)')
    ax1.set_ylabel('Flux (ppm)')
    ax2.set_ylabel('Flux (ppm)')
    
    ax1.tick_params(direction='out', which='both', length=4) 
    ax1.tick_params( which='major', pad=7, length=6,labelsize='12')
    ax2.tick_params(direction='out', which='both', length=4) 
    ax2.tick_params( which='major', pad=7, length=6,labelsize='12')

    if PARAM['Save_plots']:        
        if not os.path.exists(PARAM['Plots_save_path']):
            os.makedirs(PARAM['Plots_save_path'])
            
        fig.savefig(os.path.join(PARAM['Plots_save_path'], PARAM['ID'] + '_LC.png'))    
        
    
    if PARAM['Show_plots']:
        plt.show()
    else:
        plt.close('all')
    

    
#==============================================================================
#     
#==============================================================================

def simulate(PARAM):
    
    PARAM['LOGG'] = None
    PARAM['E_LOGG'] = None
    PARAM['TEFF'] = None
    PARAM['E_TEFF'] = None
    
    # Sysa = Additive systematics
    # Sysm = Multiplikative systematics
    # Sysf = Focus change systematics
    Time, Quality, PARAM, Sysa, Sysm, Sysf = time_vector(PARAM['CAD'], PARAM['ELAT'], verbose=False, seed=PARAM['Tseed'], PARAM=PARAM)
    random.seed(a=int(time.time()*1e6)) # Change fixed seed from time vector generation
    
    Flux = np.zeros_like(Time, dtype='float64')
    
    PARAM['Obs_duration'] = round(Time[-1] - Time[0], 2)
    
    
    if not 'batch_name' in PARAM.keys():
        PARAM['batch_name'] = ''
        
        
        
   #         .
   #         .
   #         .
   # Calculated stellar signals in relative flux and add to "Flux"  
   #         .
   #         .
   #         .
        
        
        

    if PARAM['Sim_Category'] is 'Constant':
        print('Making a constant star')
        PARAM['LOGG'] = None
        PARAM['E_LOGG'] = None
        PARAM['TEFF'] = int(np.random.uniform(4000,8000))
        PARAM['E_TEFF'] = 200
        

    #==============================================================================
    # Add noise and systematics    
    #==============================================================================
    
    # Remove flux vals from bad quality times
    idx_nan = ((Quality & 2+4+8+32+256) != 0)
    Flux[idx_nan] = np.nan
        
    # Estimate mean flux level    
    Mean_level = mean_flux_level(PARAM['TMAG'], PARAM['TEFF'])


    # Estimate shot+read+zodiacal+systematic jitter noise
    Noise_vals, PARAM = phot_noise(PARAM['TMAG'], PARAM['TEFF'], PARAM['CAD'], PARAM)    
    shot_noise_val, zodiacal_noise_val, read_noise_val, systematic_noise_val = Noise_vals
    
    shot_noise = Sysf*shot_noise_val
    zodiacal_noise = Sysm*zodiacal_noise_val
    Noise_rms = np.array([np.sqrt(np.sum(read_noise_val**2 + systematic_noise_val**2 + shot_noise[i]**2 + zodiacal_noise[i]**2))*1e-6 for i in range(len(shot_noise))])
    

    # generate white noise with seed
    PARAM['Noise_seed'] = int(time.time()*1e6)
    random.seed(a=PARAM['Noise_seed'])
    Noise =  np.array([random.normalvariate(0, Noise_rms[i]) for i in range(len(Noise_rms))])
   
    Noise_rms_simple = np.sqrt(np.sum(Noise_vals**2))
    
    # Co-add components
    Clean_flux = (Flux+1) * Mean_level 
    Noisy_flux = (Noise+Flux+1) * Mean_level 
    
    Noise_sys = Mean_level*(1+Flux+Sysa) 
    Noisy_sys_flux = Noise_sys*Sysf*Sysm + Mean_level*Noise 
    
    
    PARAM['Mean_level'] = Mean_level
    PARAM['Shot_noise'] = Noise_vals[0]
    PARAM['read_noise'] = Noise_vals[1]
    PARAM['zodiacal_noise'] = Noise_vals[2]
    PARAM['systematic_noise'] = Noise_vals[3]
    PARAM['rms_noise_simple'] = Noise_rms_simple
    
    
    return Time, Clean_flux, Noisy_flux, Noisy_sys_flux, Quality, PARAM
	


#==============================================================================
# 
#==============================================================================

def Sim_batch(No=1000, startno=1, Probs=None, data_path=None, lc_path=None, plot_path=None, batch_name=None, Tmmin=17, Tmmax=3, showplot=True, saveplot=True, savelc=True, plotps=True):
    
    if batch_name is None:
        batch_name = 'BatchX'
    if data_path is None:
        data_path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/'
    if lc_path is None:
        lc_path = 'Simulated_LCs'
    if plot_path is None:
        plot_path = 'Simulated_LC_plots'
                
    if Probs is None:        
        Probs = {'Classical_AF': 0.2,
                 'Classical_OB': 0.2,
                 'LPV': 0.05,
                 'RRLyr_Cepheid': 0.1,
                 'Solarlike': 0.15,
                 'Planet': 0.1,
                 'EB': 0.1,
                 'Constant': 0.1}
    
    cdf = [Probs.values()[0]]
    for i in xrange(1, len(Probs.values())):
        cdf.append(cdf[-1] + Probs.values()[i])

    no = startno-1
    while no<No:
        
        print('****************************************************************************')
        print(' ********************** SIMULATING STAR ' + str(no+1) + ' of ' + str(No) + ' ***************************')
        print('****************************************************************************')

        try:
            no += 1
            PARAM = {} 
            PARAM['data_path'] = data_path
            PARAM['batch_name'] = batch_name
            PARAM['LC_save_path'] = os.path.join(PARAM['data_path'], lc_path, PARAM['batch_name'])
            PARAM['Save_plots'] = saveplot
            PARAM['Save_LC'] = savelc
            PARAM['Plots_save_path'] = os.path.join(PARAM['data_path'], plot_path, PARAM['batch_name'])
            PARAM['Plot_ps'] = plotps
            PARAM['Show_plots'] = showplot

            # Calculate class based on probabilities
            random.seed(a=int(time.time()*1e6))
            r = random.random()
            idx = bisect(cdf,r)
            Sim_Category = Probs.keys()[idx]
            
            PARAM['Sim_Category'] = Sim_Category
            PARAM['Tseed'] = 15
            PARAM['ID'] = 'Star' + str(no)
            PARAM['CAD'] = 1800
            PARAM['TMAG'] = round(random.uniform(Tmmin, Tmmax), 1)
            
            PARAM['ELAT'] = 90 - np.arccos(2*random.uniform(0,1)-1)*180/np.pi  
            while PARAM['ELAT']<6:
                
                if PARAM['ELAT']<0:
                    PARAM['ELAT'] = np.abs(PARAM['ELAT'])
                else:    
                    PARAM['ELAT'] = 90 - np.arccos(2*random.uniform(0,1)-1)*180/np.pi  
                
                                 
            PARAM['TEFF'] = None
            
            T, C, N, NS, Q, PARAM = simulate(PARAM)
            DATA = np.column_stack((T, C, N, NS, Q))
        
            pprint.pprint(PARAM)
            save_LC_data(DATA, PARAM)
            plot_LC_data(T, N, C, NS, PARAM)   
        except Exception as e: 
            print str(e)
            print('**************************FAILED********************')
            pprint.pprint(PARAM)
            no -= 1
            continue


#==============================================================================
# 
#==============================================================================

def vals_from_batch(path, save_path, batch_name, ext='*.json', name_ext=''):
    
    json_files = np.array(glob.glob(os.path.join(path, ext)))
    
    Values = np.zeros(len(json_files), dtype=[('var1', 'S8'), ('var2', float), ('var3', int), ('var4', float), ('var5', float)
    , ('var6', float), ('var7', int), ('var8', int), ('var9', float), ('var10', float), ('var11', 'S40')])
    
    
    numbers = np.array([], dtype=int)
    for i in range(len(json_files)):
        file = json_files[i]
        print file
        PARAM = json.loads(open(file).read())
        numbers = np.append(numbers, int(PARAM['ID'].strip('Star')))

        Values['var1'][i] = str(PARAM['ID'])
        Values['var2'][i] = round(PARAM['TMAG'], 2)
        Values['var3'][i] = int(PARAM['CAD'])
        Values['var4'][i] = round(PARAM['Obs_duration'], 2)
        Values['var5'][i] = round(PARAM['ELAT'], 2)
        Values['var6'][i] = round(PARAM['ELON'], 2)
        try:
            Values['var7'][i] = int(PARAM['TEFF'])
            Values['var8'][i] = int(PARAM['E_TEFF'])
        except:
            Values['var7'][i] = -999
            Values['var8'][i] = -999    
        try:
            Values['var9'][i] = round(PARAM['LOGG'], 2)
            Values['var10'][i] = round(PARAM['E_LOGG'], 2)
        except:
            Values['var9'][i] = -999
            Values['var10'][i] = -999
        
        try:
            if 'Solar-like' in PARAM['Pulation_type']:
                PARAM['Pulation_type'] = 'Solar-like'
                        
            if (PARAM['Sim_Category']=='Planet'):
                Values['var11'][i] = 'Transit'
            elif (PARAM['Sim_Category']=='EB'):   
                Values['var11'][i] = 'Eclipse'
            else:   
                Values['var11'][i] = PARAM['Pulation_type']
                
    
            if 'EB_file' in PARAM.keys():
                Values['var11'][i] += ';Eclipse'
            if 'Transit_simID' in PARAM.keys():
                Values['var11'][i] += ';Transit' 
                
                if PARAM['Planet_MMR'] == 'true':
                    Values['var11'][i] += ';MMR'
                if 'Simulated_multi' in PARAM['System info file:']:
                    Values['var11'][i] += ';multi'
        
#            if 'System info' in PARAM.keys():
#                if 'EB_file' in PARAM.keys():
#                    Values['var11'][i] = PARAM['Pulation_type'] + ';Eclipse'
#                if 'Transit_simID' in PARAM.keys():
#                    Values['var11'][i] = PARAM['Pulation_type'] + ';Transit'
#            else:                                            
                
        except:
            if (PARAM['Sim_Category']=='Constant'):
                Values['var11'][i] = 'Constant'
            
            if (PARAM['Sim_Category']=='Planet'):
                Values['var11'][i] = 'Transit'
                if PARAM['Planet_MMR'] == 'true':
                    Values['var11'][i] += ';MMR'
                if 'Simulated_multi' in PARAM['System info file:']:
                    Values['var11'][i] += ';multi'
            if (PARAM['Sim_Category']=='EB'):   
                Values['var11'][i] = 'Eclipse'
#            else:     
#                Values['var11'][i] = 'Unknown'

        if 'Nflares' in PARAM.keys():
            Values['var11'][i] += ';Flare'

        if 'Spot_parameters' in PARAM.keys():
            Values['var11'][i] += ';Spots'    


    # Save results into data file
    header = 'Results file for TESS LC simulation ' + batch_name
    header += \
    """
    Column 1: ID
    Column 2: TESS magnitude
    Column 3: Cadence (sec)
    Column 4: Observing duration (days)
    Column 5: Elicptic latitude (deg)
    Column 6: Ecliptic longitude (deg)
    Column 7: Teff (K)
    Column 8: Teff error (K)
    Column 9: logg (cgs)
    Column 10: logg error (cgs)
    Column 11: Type (may include subtype separated by ;)
    """

    idx = np.argsort(numbers)
    fmt = '%s, %1.2f, %i, %1.2f, %1.2f, %1.2f, %i, %i, %1.2f, %1.2f, %s'
    np.savetxt(os.path.join(save_path, 'Data_'+batch_name+name_ext+'.txt'), Values[idx], fmt=fmt, header=header)    


    # plot targets on sphere
    Theta = (90 - Values['var5'][:])*np.pi/180
    Phi = Values['var6'][:]*np.pi/180
    phi = np.linspace(0, np.pi, 130)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    
    xi = 1.05* np.sin(Theta) * np.cos(Phi) 
    yi = 1.05* np.sin(Theta) * np.sin(Phi)
    zi = 1.05* np.cos(Theta)
        
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':0.5})
    ax.plot_wireframe(x, y, z, color='0.5', rstride=10, cstride=10)
    ax.scatter(xi, yi, zi, s=30, c='r', zorder=10)
    fig.savefig(os.path.join(save_path, 'Sky_'+batch_name+'.png'))
    plt.close('all')
    
    
    
def vals_from_batch_noisy(save_path, batch_name):
    
    
    file_name = 'Data_'+batch_name+'.txt'
    file_name_new = 'Data_'+batch_name+'_noisy.txt'
    mapping_file = os.path.join(save_path, batch_name, 'ID_mapping.txt')        
        
    # Save results into data file
    header = 'Results file for TESS LC simulation shuffled noisy data in ' + batch_name
    header += \
    """
    Column 1: ID
    Column 2: TESS magnitude
    Column 3: Cadence (sec)
    Column 4: Observing duration (days)
    Column 5: Elicptic latitude (deg)
    Column 6: Ecliptic longitude (deg)
    Column 7: Teff (K)
    Column 8: Teff error (K)
    Column 9: logg (cgs)
    Column 10: logg error (cgs)
    Column 11: Type (may include subtype separated by ;)
    """

    Data = np.genfromtxt(os.path.join(save_path, file_name), dtype=None, delimiter=',')
    Data_new = copy.deepcopy(Data)
    
    Mapping = np.loadtxt(mapping_file, dtype=int)
    numbers = np.array([])
    for i in range(len(Data)):
        N = int(Data[i][0].strip('Star'))
        idx = (Mapping[:,0]==N)
        Data_new[i][0] = 'Star' + str(int(Mapping[idx,1][0]))
        numbers = np.append(numbers, int(Mapping[idx,1][0]))

    idx_sort = np.argsort(numbers)
    
    Data_new = Data_new[idx_sort]
    fmt = '%s, %1.2f, %i, %1.2f, %1.2f, %1.2f, %i, %i, %1.2f, %1.2f, %s'
    np.savetxt(os.path.join(save_path, file_name_new), Data_new, fmt=fmt, header=header)    


    
#==============================================================================
# 
#==============================================================================

def zipdir(path, ziph, ext='.npy'):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        files2 = [f for f in files if f.endswith(ext)]
        print ext
        print files2
        for file in files2:
            filepath = os.path.join(root, file)
            ziph.write(filepath, os.path.basename(filepath))

def pack_files(folder, folder_name, ext='npy'):  
    zipfolder = os.path.join(folder, folder_name + '.zip')
    zipf = zipfile.ZipFile(zipfolder, 'w', zipfile.ZIP_DEFLATED)
    zipdir(folder, zipf, ext)
    zipf.close()
    
#==============================================================================
#         
#==============================================================================

if __name__ == "__main__":
    
    Simulate_single = False
    Simulate_batch = True


    if Simulate_batch:

        Probs = {'Classical_AF': 0,
                 'Classical_OB': 0,
                 'LPV': 0,
                 'RRLyr_Cepheid': 0,
                 'Solarlike': 0,
                 'Planet': 0,
                 'EB': 0,
                 'Constant': 1}         
                 
        batch_name = 'Crude_corr_test'
        lc_path = 'Simulated_LCs'
        plot_path = 'Simulated_LC_plots' 
        data_path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/'
        Sim_batch(No=10, startno=0, Probs=Probs, lc_path=lc_path, plot_path=plot_path, batch_name=batch_name, Tmmin=10, Tmmax=1, showplot=False, saveplot=True, savelc=True, plotps=False)
        
        sys.exit()
        
        
        # Create values file
        local_path = ''
        path_to_data = os.path.join(local_path, lc_path, batch_name)
        
        save_path = os.path.join(local_path, lc_path)
        vals_from_batch(path_to_data, save_path, batch_name, ext='*.json', name_ext='')   
        
        # Pack files into zip files
        pack_files(path_to_data, 'noisy_files', ext='.noisy')   
        pack_files(path_to_data, 'clean_files', ext='.clean')   
        pack_files(path_to_data, 'json_files', ext='.json')   
    
    
    if Simulate_single:      
#        Sim_Category = 'Classical_AF'
    #    Sim_Category = 'Classical_OB'
#        Sim_Category = 'LPV'
#        Sim_Category = 'RRLyr_Cepheid'
#        Sim_Category = 'Solarlike'
#        Sim_Category = 'Planet'
        Sim_Category = 'Constant'
#        Sim_Category = 'EB'
       
        CAD = 1800
        TMAG = 10
        ELAT = 30
        ID = 'star1'
        PARAM = {}
        
        PARAM['ID'] = ID
        PARAM['data_path'] = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/'
        PARAM['LC_save_path'] = os.path.join(PARAM['data_path'], 'Simulated_LCs')
        PARAM['CAD'] = CAD
        PARAM['TMAG'] = TMAG
        PARAM['ELAT'] = ELAT
        PARAM['Sim_Category'] = Sim_Category
        PARAM['Tseed'] = 15
        #PARAM['RG_or_MS'] = 0.5#0.5 # Probability of RG vs MS
        PARAM['Planet_chance'] = 0.2
        PARAM['ttv_prob'] = 0.2
        PARAM['Planet_chance_mmr'] = 1
        PARAM['Allow planet'] = False
        PARAM['Single_ov_multi_plan'] = 0.2 # Probability of Single vs. multi planet (no TTV)
        PARAM['planets_max'] = 5
        PARAM['Save_plots'] = True
        PARAM['Save_LC'] = True
        PARAM['save_trans_all'] = False
        PARAM['save_trans_info'] = True
        PARAM['save_flare'] = True
        PARAM['Plots_save_path'] = os.path.join(PARAM['data_path'], 'Simulated_LC_plots')
        PARAM['Plot_ps'] = True
        PARAM['Show_plots'] = True
        PARAM['Save_spots'] = True
        
        PARAM['EB_dilute_chance'] = 0.5
        PARAM['Allow EB'] = False
        PARAM['EB_chance'] = 0.8
        
        T, C, N, NS, Q, PARAM = simulate(PARAM)
        DATA = np.column_stack((T, C, N, NS, Q))
        pprint.pprint(PARAM)        
        save_LC_data(DATA, PARAM)        
        plot_LC_data(T, N, C, NS, PARAM)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    