"""
04 - Time traces and normalization
This script calculates the normailzed time traces of selected peaks.

INPUTS: Config file
"""

# %%

import os
import sys
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

config_file = "config_2020_19_07.cfg"

import tqdm

import glob
import numpy as np
import pandas as pd
import skued
import scipy.io as spio
import pylab as plt

import helpers.fedutils as utils


# Load Config

PATH_CFG = os.path.join(DATA_DIR, "config_2020_19_07.cfg")
dict_path, dict_numerics = utils.read_cfg(PATH_CFG)
LOG = os.path.join(DATA_DIR, dict_path['path_log'])
PATH = os.path.join(DATA_DIR, dict_path['path'])

partial = dict_numerics['partial']
if partial == 1:
    diffraction_images = glob.glob(os.path.join(PATH, "*.tif"))[: dict_numerics["to_file"]]
else:
    diffraction_images = glob.glob(os.path.join(PATH,'*.tif'))
number_of_diffraction_images = len(diffraction_images)
print(f"Number of diffraction images used: {number_of_diffraction_images}")

# %%
# Loads flatfield
PATH_FF =  os.path.join(DATA_DIR, dict_path['path_ff'])
FFmat = spio.loadmat(PATH_FF)
FF = FFmat['FF']

# Loads initial peak positions
peakpos_all = pd.read_csv(os.path.join(DATA_DIR, "peak_positions.csv"))
number_of_peaks = np.shape(peakpos_all)[0]

# File is an output of script 02. Contains evolution of center and residuals.
# Only first center position is ectracted.
CRPATH = dict_path['path_center']
cen_res = np.loadtxt(os.path.join(DATA_DIR, CRPATH),dtype = float, skiprows = 1)
center=cen_res[0,0:2]

# Loads laser background
PATH_BKG = dict_path['path_bkg']
LASER_BKG = np.array(skued.diffread(os.path.join(DATA_DIR, PATH_BKG)), dtype = np.float64)

# %%
# Loads parameters
window_size_intensity = dict_numerics['window_size_intensity']
window_size_background_peaks= dict_numerics['window_size_background_peaks']
mask_size_zero_order = dict_numerics['mask_size_zero_order']
max_distance = dict_numerics['max_distance']

# Creating dummy image for displaying masks etc
dummy =np.array(skued.diffread(diffraction_images[1]), dtype = np.int64)
dummy_bkg = utils.remove_bgk(dummy, LASER_BKG, FF)

# %%
# Loads masks
path_mask = os.path.join(DATA_DIR, dict_path['path_mask'])
mask_total = spio.loadmat(path_mask)['mask_total']
mask_zero_order = utils.mask_image(dummy.shape, [center], [mask_size_zero_order])*mask_total
masked_bragg = utils.mask_image(dummy.shape, peakpos_all, window_size_intensity*np.ones(number_of_peaks))*mask_zero_order
masked_total_counts = utils.mask_image(dummy.shape, [center], [max_distance], True)*mask_zero_order
masked_dyn_bg = utils.mask_image(dummy.shape, peakpos_all, window_size_background_peaks*np.ones(number_of_peaks))#*mask_zero_order

# Loads peak positions
PATH_PEAKPOS = dict_path['peak_pos']
# Checks if file exists
exists = os.path.isfile(PATH_PEAKPOS)

if not exists:
    print('First need to generate peak position file...')
    PEAK_POS = utils.peakpos_evolution(diffraction_images, mask_total, LASER_BKG, FF, peakpos_all, dict_numerics['lens_corr_repetitions'],dict_numerics['lens_corr_window_size'])
    np.savetxt(PATH_PEAKPOS,
               PEAK_POS, header='No peaks)')
    PEAK_POS = PEAK_POS.reshape((number_of_diffraction_images,number_of_peaks, 2))
    dict_numerics['calculate_peak_evolution'] = 0
else:
    if exists:
        PEAK_POS = np.loadtxt(PATH_PEAKPOS)

# Path for saving output file
PATH_OUTPUT = PATH + dict_path['path_output'] + '/output_norm04_' + PATH_CFG.split('/')[-1].split('.')[0] + '.txt'

# %% Display Masks
# Display images with masks
cmap = 'inferno'
b = 2
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), sharex = True, sharey = True)
ax1.pcolormesh(dummy,vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)
ax2.pcolormesh(dummy_bkg,vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)
ax3.pcolormesh(dummy_bkg*b,vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)

ax4.pcolormesh(dummy_bkg*masked_bragg,vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)
for idp, pp in enumerate(peakpos_all):
    ax4.plot(pp[1] ,pp[0],  marker = 'x', markersize = 4, color = 'r')
    ax4.text(pp[1]  - 10, pp[0] -10, str(idp), color = 'r')
ax5.pcolormesh(dummy_bkg*masked_total_counts, vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)
ax6.pcolormesh(dummy_bkg*masked_dyn_bg, vmin = -0,vmax = 0.1*np.nanmax(np.abs(dummy)), cmap = cmap)

for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

ax1.set_title("Original")
ax2.set_title("Laser background removed")
ax3.set_title("Background without_laser")
ax4.set_title("Bragg peaks windows")
ax5.set_title("Mask for total e counts")
ax6.set_title("Mask for dynamical background")
lim = 800
# plt.xlim(center[1][0] - lim, center[1][0] + lim)
# plt.ylim(center[0][0] - lim, center[0][0] + lim)
plt.tight_layout()
plt.show()
plt.savefig("//nap1.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/Timetraces/masks.png", format="png")

# =============================================================================
# START ANALYSIS
# Old data is loaded when not deleted.
# =============================================================================

# Get tif files from data set dir
fn=glob.glob(PATH+'*.tif')        #get file list
num_img=len(fn) 
for k in range(0,len(fn)):
    fn[k] = os.path.basename(fn[k])

if os.path.isfile(DATA_DIR+'output_data_04.npz')==True:
    npzfile = np.load(DATA_DIR+'output_data_04.npz')
    total_counts = npzfile['tc']
    peakpos_evolution = npzfile['pp_evo']
    intensities_raw = npzfile['int_raw']
    npzfile.close()
    
else:
    #Loop over all images collecting peak intensities, total counts etc....
    if partial == 0:
        partial=len(fn)
    peakpos_evolution = []
    total_counts = np.zeros(len(fn[0:partial]))
    background_raw = np.zeros(len(fn[0:partial]))
    #distance_matrix = centeredDistanceMatrix(np.shape(dummy)[0])
    intensities_raw = np.zeros((number_of_peaks, len(fn[0:partial])))
    
    for idx, bn in tqdm.tqdm(enumerate(fn[0:partial])):
        image = np.array(skued.diffread(PATH + bn), dtype = np.int64)
        #checks for saturation
        if np.nanmax(np.nanmax(image))==65000:
            print('Warning: Image '+ str(k)+' is saturated!')
            
        #Apply mask
        image = image*mask_total
        #Substract background and flatfield
        image = utils.remove_bgk(image, LASER_BKG, FF)
        new_peakpos_all = utils.refine_peakpos_arb_dim(peakpos_all, image, 0, 8)
        peakpos_evolution.append(new_peakpos_all)
        peakpos_all = new_peakpos_all
        #Calculates total electrons number
        total_counts[idx] = np.nansum(image*masked_total_counts)
        
        image_bgs = image
        for idp, peak in enumerate(peakpos_all):
            intensities_raw[idp, idx] = utils.sum_peak_pixels(image_bgs, peak, window_size_intensity)
            
    #Saving data in an npz file
    np.savez(DATA_DIR+'output_data_04.npz',pp_evo=peakpos_evolution,\
             int_raw=intensities_raw,tc=total_counts)

# =============================================================================
# DISPLAY DATA
# =============================================================================
# Display total counts and peakpos evolution
plt.figure()
plt.plot(np.arange(len(total_counts)),total_counts)
plt.xlabel('Image number [#]')
plt.ylabel('Total counts [#]')
plt.show()

# =============================================================================
# NORMALIZATION
# =============================================================================
# Normalize dataset via minimizing correlation of NOE
offset, corr_int = utils.normalize_pearson(np.mean(intensities_raw,axis=0), total_counts, tolerance = 1e-15, max_steps = 10000)

# Normalize peak intensities
intensities_norm=[]
for k in range(0,np.size(intensities_raw,axis=0)):
    intensities_norm.append(np.divide(intensities_raw[k,:],total_counts-offset))
intensities_norm = np.array(intensities_norm,dtype=float)

# Read log file. Get delays, filenames and scans
with open(PATH + LOG, 'rt') as meta:
    lines = meta.readlines()
del lines[0]

fn = []
delays = []
scans = []
for line in lines:
    fn.append(line.split('\t')[0] + '.tif')
    delays.append(float(line.split('\t')[3]))
    scans.append(int(line.split('\t')[2]))
delays = np.array(delays,dtype=float)

no_scans = np.unique(scans)
no_delays = len(np.unique(delays))
number_of_peaks = intensities_norm.shape[0]

# Sort delays and intensities
delay_sort, ind = np.unique(delays, return_inverse=True)
intensities_mean_norm = np.empty((intensities_norm.shape[0],0), dtype=float)

# Summarize intensities
for ii in range(0,len(delay_sort)):
    temp_ind=np.where(ind==ii)
    intensities_mean_norm = np.hstack((intensities_mean_norm,\
    np.mean(intensities_norm[:,temp_ind[0]],axis=1).reshape(number_of_peaks,1)))
    #binned_int_err(ii,:)=std(fit_int_norm(temp_ind,:),1)

total_int = np.mean(intensities_mean_norm,axis=0)

# =============================================================================
# Display intensity evolution
# =============================================================================

plt.figure()
plt.plot(delays,intensities_raw[8,:],'o')
plt.show()

plt.figure()
for i in range(0,number_of_peaks):
    plt.plot(delay_sort/1000,intensities_mean_norm[i,:]/intensities_mean_norm[i,0])
plt.xlabel('Delay $\Delta t$ [ps]')
plt.ylabel('Relative Intensity $\Delta I_r$ [%]')
plt.savefig("//nap1.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/Timetraces/timetraces.png", format="png")
plt.show()

plt.figure()
plt.plot(delay_sort/1000,total_int/total_int[0])
plt.xlabel('Delay $\Delta t$ [ps]')
plt.ylabel('Mean total intensity $\Delta I_r$ [%]')
plt.savefig("//nap1.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/Timetraces/total_int.png", format="png")
plt.show()

# =============================================================================
# SAVE DATA
# =============================================================================

# Saving data in an npz file
np.savez(DATA_DIR+'output_data_04_02_norm.npz',int_norm=intensities_norm,\
         int_mean_norm=intensities_mean_norm,tot_int=total_int,delay=delay_sort,\
        delay_raw=delays,offset=offset,tc=total_counts)
