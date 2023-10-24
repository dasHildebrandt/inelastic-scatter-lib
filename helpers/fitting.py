"""
Created on Tue May  5 23:36:58 2020

@author: Patrick Hildebrandt

Routine applies center of mass (COM) method and 2D-pseudovoigt fitting on a single 
image. COM and fit data could be used to estimate the conversion factor between
reciprocal and pixel space.
"""

import sys, configparser
from PIL import Image
import numpy as np
import pylab as plt
import tqdm
import scipy.optimize as opt
import os
import math as m
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from inPho_mod import center_of_mass as com


def pseudoVoigt2D_surfBG(
        xdata_tuple: tuple,
        amp: float,
        etax: float,
        x0: float, 
        sx: float,
        y0:float, 
        sy: float, 
        BGA1: float, 
        BGA2: float, 
        BGb: float
):
    """Fit function of a two dimensional Pseudo-Voigt function """
    (xgrid, ygrid) = xdata_tuple
    z = amp*( (1-etax) * np.exp( -np.log(2)*(np.power(((xgrid-x0)/(2*sx)),2) + \
        np.power(((ygrid-y0)/(2*sy)),2) )) + etax/(1 + np.power(((xgrid-x0)/(2*sx)),2) \
        + np.power(((ygrid-y0)/(2*sy)),2)))
    z = z + BGA1*xgrid + BGA2*ygrid + BGb
    return z.ravel()

#This is a function I used to visualize my fit quality.

def plot_comp_fit_exp(x, y, peak1, z, save = False, savename = '', path_fits = 'my_path'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize = (6, 4))

    ax1.set_title('Experiment')
    pcm1 = ax1.pcolormesh(x, y, peak1, vmin = 0, vmax = np.max(z), cmap = 'viridis')
    ax1.get_xaxis().set_ticks([])
    ax2.set_title('Theory')
    pcm2 = ax2.pcolormesh(x, y, z , vmin = 0, vmax = np.max(z), cmap = 'viridis')
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    lim2 = np.max(np.abs(peak1- z))
    ax3.set_title('Diff')
    diff = ax3.pcolormesh(x, y, peak1 - z, cmap = 'seismic', vmin = -lim2, vmax = +lim2)
    ax3.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    plt.colorbar(diff, ax=ax3)
    plt.colorbar(pcm2, ax=ax2)
    plt.colorbar(pcm1, ax=ax1)
    ax4.set_title('slices (integrated)')
    ax4.plot(np.sum(peak1, axis = 0), color = 'r')
    ax4.plot(np.sum(z, axis = 1), color = 'k') # - 0.17)

    ax4.plot(np.sum(peak1, axis = 1), color = 'r')
    ax4.plot(np.sum(z, axis = 0), color = 'k') # - 0.17)

    plt.subplots_adjust(left = 0)
    
    print(os.path.exists(path_fits))
    if not os.path.exists(path_fits):
        os.makedirs(path_fits)
    if save == True:
        plt.savefig('fits/' + savename + '.png')
    plt.show()

#Prepare txt files for saving positions.
    
#Write x and y parameters with covarinaz in txt
comFile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/CenterOfMass_data.txt'
comtxt=open(comFile,'w')
comtxt.writelines('x0'+'\t'+'y0'+'\n')

#Write x and y parameters with covarinaz in txt
tFile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/fit_data.txt'
txt=open(tFile,'w')
txt.writelines('x0'+'\t'+'dx'+'\t'+'y0'+'\t'+'dy'+'\n')

#Loading image with BG FF
#tif format
imgfilename='C:/Users/patri/Documents/Topfloor/20190502/meas2/meas2_0015.tif'
#tif format
BGfile='C:/Users/patri/Documents/Topfloor/20190502/meas2/laser_background/meas1_0039.tif'
#mat file
FFfile='C:/Users/patri/Documents/Topfloor/20180808_flatfield_improved_without_bad_pixel_mask.mat'
#FFfile='C:/Users/patri/Documents/Topfloor/20180808_flatfield_improved_with_strict_bad_pixel_mask.mat'
#txt file peak selection
Sfile='//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/peak_selection_large.txt'

#Apply corrections to image
img=Image.open(imgfilename)
IM = np.asarray( img, dtype="float64" )
if BGfile:
    BG=Image.open(BGfile)
    BGimg = np.asarray(BG, dtype="float64" )
    IMcor=IM-BGimg
if FFfile:
    FFmat=sio.loadmat(FFfile)
    FFimg=FFmat['FF'] #changes if FF file chnges
    IMcor=np.multiply(FFimg,IMcor)
    
txtPS = open(Sfile,'r')
ps=txtPS.readlines()
txtPS.close()

for jj in range(1,len(ps)):
    data=ps[jj].split('\t') #read every line from txt, split columns by tabs
    
    x0     = float(data[0]) #position x of peak idx. 
    y0     = float(data[1]) #position y of peak idy
    
    ws = int(float(data[3]))
    mil = data[2]
    
    #Defines the boundaries of the fits for chosen peak
    xlb = int(x0) - ws
    xub = int(x0) + ws
    ylb = int(y0) - ws
    yub = int(y0) + ws
    
    x = np.arange(xlb, xub, 1)
    y = np.arange(ylb, yub, 1)
    
    peak = IMcor[xlb:xub,ylb:yub]
    
    plt.figure()
    plt.imshow(peak) 
    plt.show()  
    
    xpos,ypos=com(peak,x,y)
    print(str(xpos)+'  '+str(ypos))
    
    comtxt.writelines(str(xpos)+'\t'+str(ypos)+'\n')
comtxt.close()
    

paras=[]
for j in range(1,len(ps)):
    data=ps[j].split('\t') #read every line from txt, split columns by tabs

#You need to create a vector containing all the x positions of the peaks you wish to fit.
#This could be the output of a previous script that you wrote for the user to pick peaks in an image.

    x0     = float(data[0]) #position x of peak idx. 
    y0     = float(data[1]) #position y of peak idy
    #print(x0)
    #print(y0)

    #You also need to create a vector "ROI_BG" containing the widths of the ROI you want to fit. This might be peak dependent.
    ws = int(float(data[3]))
    mil = data[2]
    #print(ws)
    #print(mil)
    
    #Defines the boundaries of the fits for chosen peak
    xlb = int(x0) - ws
    xub = int(x0) + ws
    ylb = int(y0) - ws
    yub = int(y0) + ws
    
    x = np.arange(xlb, xub, 1)
    y = np.arange(ylb, yub, 1)
    
    xgrid, ygrid = np.meshgrid(x, y)
    
    #Here you will need to insert some code where you load the image you want to fit. The whole diffraction image at this point.
    #It should be an image with FF, BG correction already done, and probably the average image over the different scans.
    
    peak = IMcor[xlb:xub,ylb:yub]
# =============================================================================
#     plt.imshow(peak, cmap=cm.Blues)
#     plt.show()
# =============================================================================
    
    
    initial_guess = (10000, 0.1, x0, 8, y0, 8, 0.1, 0.1, 200)
    fitbounds=([500, 0, x0-10, 2, y0-10, 2, -150, -150, 0],
               [100000, 1, x0+10, 25, y0+10, 25, 150 ,150, 2000])                                                            

    popt, pcov = opt.curve_fit(pseudoVoigt2D_surfBG, (xgrid, ygrid), peak.ravel(), \
                p0 = initial_guess,check_finite=False,bounds=fitbounds)
    sdiv=np.sqrt(np.diag(pcov))
    
    data_fitted = pseudoVoigt2D_surfBG((xgrid, ygrid), *popt)
    z = data_fitted.reshape(2*ws,2*ws) 
    paras.append(popt)
    
#Test function here
    plot_comp_fit_exp(x, y, peak, z, save = False, savename = 'test')
    
    txt.writelines(str(popt[2])+'\t'+str(sdiv[2])+'\t'+str(popt[4])+'\t'+str(sdiv[4])+'\n')

txt.close()

       

