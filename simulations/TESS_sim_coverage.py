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
import sys, os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
import random
from matplotlib.patches import Polygon
from astropy import units as u
from astropy.coordinates import SkyCoord
mpl.rcParams['font.family'] = 'serif'
import scipy.interpolate as INT
from bisect import bisect

plt.ioff()

disk_path = '/media/mikkelnl/Elements/'
#disk_path = '/media/Elements/'
#===============================================================================
# Code
#===============================================================================


#def sky_coverage_simple(RA, DEC, ELAT=None):
#    
#    if not ELAT is None:
#        eclip_lat = ELAT
#    else:
#        c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
#        eclip_lat0 = c_icrs.barycentrictrueecliptic.lat.deg
#        eclip_lat = np.arcsin(np.abs(np.sin(eclip_lat0*np.pi/180)))*180/np.pi
#    
#    obs_unit = 27.4
#    r = random.random()
#    if eclip_lat<=30:
#        duration = obs_unit
#    elif (eclip_lat<=60) & (eclip_lat>30):
#        duration = obs_unit*(r>=0.3) + (2*obs_unit)*(r<0.3)
#    elif (eclip_lat<=82) & (eclip_lat>60):
#        duration = obs_unit*(r>=0.85) + (2*obs_unit)*((r<0.85) & (r>=0.55)) + (3*obs_unit)*((r<0.55) & (r>=0.45)) + (4*obs_unit)*((r<0.45) & (r>=0.4)) \
#         + (5*obs_unit)*((r<0.45) & (r>=0.4))  + (6*obs_unit)*((r<0.45) & (r>=0.4)) + (7*obs_unit)*((r<0.45) & (r>=0.4)) + (8*obs_unit)*((r<0.48) & (r>=0.34)) \
#          + (9*obs_unit)*((r<0.34) & (r>=0.22)) + (10*obs_unit)*((r<0.22) & (r>=0.12)) + (11*obs_unit)*((r<0.12) & (r>=0.05)) + (12*obs_unit)*(r<0.05)
#    else:
#        duration = 13*obs_unit
#    return duration

#==============================================================================
# 
#==============================================================================

def sky_coverage_calculated(PARAM, verbose=False):
    path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/Codes/Data/' 
    Coverage = np.loadtxt(path + 'Coverage_vs_lat.txt')
    
    if not PARAM['ELAT'] is None:
        eclip_lat = PARAM['ELAT']
    else:
        eclip_lat = 45
        
        
#    else:    
#        c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
#        eclip_lat0 = c_icrs.barycentrictrueecliptic.lat.deg
#        eclip_lat = np.arcsin(np.abs(np.sin(eclip_lat0*np.pi/180)))*180/np.pi
           
    
    obs_unit = 27.4
    r = random.random()*100
    
    lats = np.linspace(0, 90, Coverage.shape[0])
    Coverage_at_lat = np.array([])
    Coverage_durations = np.array([])
    for i in range(Coverage.shape[1]):
        Interp = INT.InterpolatedUnivariateSpline(lats, Coverage[:,i], k=1)
        if not Interp(eclip_lat)==0:
            Coverage_at_lat = np.append(Coverage_at_lat, Interp(eclip_lat))
            Coverage_durations = np.append(Coverage_durations, i)
    
    
    Coverage_durations = Coverage_durations[np.argsort(Coverage_at_lat)]
    Coverage_at_lat = Coverage_at_lat[np.argsort(Coverage_at_lat)]
    
    cdf = [Coverage_at_lat[0]]
    for i in xrange(1, len(Coverage_at_lat)):
        cdf.append(cdf[-1] + Coverage_at_lat[i])
    
    idx = bisect(cdf,r)
    duration = Coverage_durations[idx]*obs_unit
    

    
    if verbose:
        print('Ecliptic latitude', eclip_lat)
        print('Ecliptic longitude', ELON)
        print('Possible durations', Coverage_durations*obs_unit)
        print('Duration probabilities', Coverage_at_lat)
        print('Chosen duration (days): ', duration, str(Coverage_durations[idx]) + ' observing sectors')
            
    return duration, PARAM
    
#==============================================================================
# 
#==============================================================================

def calc_coverage():
    path = '/home/mikkelnl/ownCloud/Documents/Asteroseis/TESS/TDA2/Simulations/Codes/Data/'       
       
     #12.857142857142858#9.5 #12.857142857142858#9.42477796076938
    angle_seg = 360/13    
#    phis = np.array([6, ])
#    phis = np.array([6, 30, 54, 90-np.sqrt(2)*12, 90, 90+np.sqrt(2)*12])
    phis = np.array([6, 30, 54, 78, 90, 102])
    lams = np.arange(0,13*angle_seg, angle_seg)   
#    factor = np.pi/180
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lams_outer = np.arange(0,360, 1)*np.pi/180
    
    
    ax.plot(90*np.cos(lams_outer), 90*np.sin(lams_outer), 'k--')
    ax.plot(60*np.cos(lams_outer), 60*np.sin(lams_outer), 'k--')
    ax.plot(30*np.cos(lams_outer), 30*np.sin(lams_outer), 'k--')
    
#    XX = np.array([0,-12])
#    YY = np.array([0, 12])
    
    Patch_coll = []
    
    
    for i in range(len(lams)):
        
        Xs1 = np.array([])
        Ys1 = np.array([])
        Xs2 = np.array([])
        Ys2 = np.array([])
    
        lam = lams[i]
        
#        if i>0:
#            continue

#        rotMatrix = np.array([[np.cos(lam*factor), -np.sin(lam*factor)], 
#                             [np.sin(lam*factor),  np.cos(lam*factor)]])

        for j in range(len(phis)):
            
            phi = phis[j] 
            
            if phi>90:
                phi0 = 180 - phi
            else:
                phi0 = phi
                
                
            angle_at_rim = 24/(np.cos(np.radians(phi0)))/2
            if angle_at_rim>=180:
                angle_at_rim = 180
                
            w = (90-phi)           
            if j==0:
                lam_finem = np.linspace(lam-angle_at_rim, lam, 50) 
                lam_finep = np.linspace(lam, lam+angle_at_rim, 50) 
                # Edge around limb       
                x = w*np.cos(np.radians(lam_finep))
                y = w*np.sin(np.radians(lam_finep))
                y2 = w*np.sin(np.radians(lam_finem))
                x2 = w*np.cos(np.radians(lam_finem))
            else:
                # Edge around limb   
                
                if np.abs(w)==12:
                    w *= np.sqrt(2)
                    angle_at_rim=45            
                    
                x = w*np.cos(np.radians(lam+np.sign(w)*angle_at_rim))
                y = w*np.sin(np.radians(lam+np.sign(w)*angle_at_rim))                
                y2 = w*np.sin(np.radians(lam-np.sign(w)*angle_at_rim))
                x2 = w*np.cos(np.radians(lam-np.sign(w)*angle_at_rim)) 
                
                if w==0:
                    x=-12*np.sin(np.radians(lam))
                    y=12*np.cos(np.radians(lam))
                    x2=12*np.sin(np.radians(lam))
                    y2=-12*np.cos(np.radians(lam))
                
#            ax.plot(x2, y2, 'k.', lw=0.5)
#            ax.plot(x, y, 'k.', lw=0.5)
            
            Xs1 = np.append(Xs1, x)
            Xs2 = np.insert(Xs2, 0, x2)
            
            
            Ys1 = np.append(Ys1, y)
            Ys2 = np.insert(Ys2, 0, y2)
                   
#            # Edge around center          
#            xoff = (-12)*np.cos(np.radians(lam)) 
#            yoff = (-12)*np.sin(np.radians(lam))
#                    
#            cx = np.dot(rotMatrix, XX)
#            cy = np.dot(rotMatrix, YY)
#            xx = np.array([cx[0], cy[0]])+xoff
#            yy = np.array([cx[1], cy[1]])+yoff
#    
#            # Plot slab outlines
#            ax.plot(xx, yy, 'k', lw=0.5)
#            ax.plot(x, y, 'k', lw=0.5)
#            ax.plot([xx[0], x[0]], [yy[0], y[0]], 'k', lw=0.5)
#            ax.plot([xx[-1], x[-1]], [yy[-1], y[-1]], 'k', lw=0.5)
#            
        X = np.append(Xs1, Xs2)
        Y = np.append(Ys1, Ys2)
        
        # plot shaded regions
#        points = zip(x,y) + [(xx[-1], yy[-1]), (xx[0], yy[0])]
        points = zip(X, Y)
        patches = Polygon(points, facecolor='k', alpha=0.2)
        Patch_coll.append(patches)
        
        ax.add_patch(patches)
    
    ax.plot(12*np.cos(lams_outer), 12*np.sin(lams_outer), 'r:')
    ax.plot(36*np.cos(lams_outer), 36*np.sin(lams_outer), 'r:')
    ax.plot(60*np.cos(lams_outer), 60*np.sin(lams_outer), 'r:')
    ax.plot(84*np.cos(lams_outer), 84*np.sin(lams_outer), 'r:')        
    plt.axis('equal') 
    
    fig.savefig(path + 'Coverage.png')
    plt.show()
    

    #==============================================================================
    #      
    #==============================================================================
    
    ref_phis = np.arange(0,90.01,0.01)
    lams_ref = np.linspace(0, 2*np.pi, 1000)
    
    ref_points_x = np.cos(lams_ref) 
    ref_points_y = np.sin(lams_ref) 
    
    Fin_vals = np.zeros([len(ref_phis), 14])
    
    for j in range(len(ref_phis)):
        
        
        phi = 90 - ref_phis[j]        
        ref_points = zip(phi*ref_points_x, phi*ref_points_y)
        inside = np.zeros(len(lams_ref))
    
        for pat in Patch_coll:
            inside += np.array([pat.get_path().contains_point(p) for p in ref_points], dtype=int)
        
        for k in range(14):
            Fin_vals[j, k] = np.sum(inside==k)*100/len(lams_ref)
    
    
    #==============================================================================
    #      
    #==============================================================================
    
    
    fig2 = plt.figure(figsize=(15, 8))
    fig2.subplots_adjust(bottom=0.1, right=0.97, top=0.87, left=0.1)
    ax2 = fig2.add_subplot(111)
    
    cmap = plt.cm.Set1
    cNorm3  = colors.Normalize(vmin=0, vmax=13)
    scalarMap3 = cmx.ScalarMappable(norm=cNorm3,cmap=cmap)
    
    for jj in range(Fin_vals.shape[1]):
        col = scalarMap3.to_rgba(jj)
        ax2.plot(ref_phis, Fin_vals[:, jj], color=col, lw=1.5, label=str(int(jj*27.4))+' days')
           
    ax2.set_xlim([0, 90])
    ax2.set_ylim([0, 102])
    ax2.set_xlabel('Ecliptic latitude (degrees)', fontsize=16)
    ax2.set_ylabel('Chance of coverage (%)', fontsize=16)
    
    ax2.tick_params(direction='out', which='both', length=4)                     
    ax2.tick_params(which='major', pad=6, length=5)          
    
    ax2.xaxis.set_major_locator(MultipleLocator(5))                           
    ax2.xaxis.set_minor_locator(MultipleLocator(2.5))                             
    ax2.yaxis.set_major_locator(MultipleLocator(10))                            
    ax2.yaxis.set_minor_locator(MultipleLocator(5))  
    ax2.legend()   
    
    ax2.grid(True, which='major', linestyle='--', lw=0.5, color='0.4')
    ax2.grid(True, which='minor', linestyle=':', lw=0.5, color='0.6')
    ax2.set_axisbelow(True)
    
    ax2.legend(bbox_to_anchor=(0., 1.01, 1, .1), loc=3, mode='expand', ncol=7, borderaxespad=0.,borderpad=0.4,handlelength=1.5, handletextpad=0.4, scatterpoints=1, numpoints=1,fancybox=True, prop={'size':14})
    ax2.xaxis.tick_bottom()    
    
    fig2.savefig(path + 'Coverage2.png')
    
    np.savetxt(path + 'Coverage_vs_lat.txt', Fin_vals)
    
    plt.show() 
    
    
    
#==============================================================================
#         
#==============================================================================

if __name__ == "__main__":
    
#    calc_coverage()        
    
    D = sky_coverage_calculated(50, verbose=True)
   
    print np.mean(D)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    