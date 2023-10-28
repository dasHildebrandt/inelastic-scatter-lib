"""
Created on Tue May  5 23:36:58 2020

@author: Patrick Hildebrandt

Routine applies center of mass (COM) method and 2D-pseudovoigt fitting on a single 
image. COM and fit data could be used to estimate the conversion factor between
reciprocal and pixel space.
"""

from PIL import Image
import numpy as np
import pylab as plt
import scipy.optimize as opt
import os
import math as m
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from helpers.tools import center_of_mass


def pseudoVoigt2D_surfBG(
        xdata_tuple: tuple,
        amp: float,
        etax: float,
        x0: float, 
        sx: float,
        y0: float, 
        sy: float, 
        BGA1: float, 
        BGA2: float, 
        BGb: float
)-> np.ndarray: 
    """Two dimensional Pseudo-Voigt function """
    (xgrid, ygrid) = xdata_tuple
    z = amp*( (1-etax) * np.exp( -np.log(2)*(np.power(((xgrid-x0)/(2*sx)),2) + \
        np.power(((ygrid-y0)/(2*sy)),2) )) + etax/(1 + np.power(((xgrid-x0)/(2*sx)),2) \
        + np.power(((ygrid-y0)/(2*sy)),2)))
    z = z + BGA1*xgrid + BGA2*ygrid + BGb
    return z.ravel()


def plot_comp_fit_exp(x, y, peak1, z, save = False, savename = '', path_fits = 'my_path'):
    """This is a function I used to visualize my fit quality."""
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

