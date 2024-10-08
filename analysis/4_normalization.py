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
import ast
import glob
import numpy as np
import pandas as pd
import skued
import scipy.io as spio
import matplotlib.pyplot as plt
from PIL import Image

from helpers.fedutils import (
    get_mask_image,
    remove_bgk,
    read_cfg,
    sum_peak_pixels,
    get_peak_position_evolution,
    refine_peak_positions,
    normalize_pearson,
)
from helpers.tools import correct_image


# Load Config

configuration_file = os.path.join(DATA_DIR, "config_2020_19_07.cfg")
dict_path, dict_numerics = read_cfg(configuration_file)
log_file = os.path.join(DATA_DIR, dict_path["path_log"])
diffraction_data_path = os.path.join(DATA_DIR, dict_path["path"])

partial = dict_numerics["partial"]
if partial == 1:
    diffraction_images = glob.glob(os.path.join(diffraction_data_path, "*.tif"))[
        : dict_numerics["to_file"]
    ]
else:
    diffraction_images = glob.glob(os.path.join(diffraction_data_path, "*.tif"))
number_of_diffraction_images = len(diffraction_images)
print(f"Number of diffraction images used: {number_of_diffraction_images}")

flatfield_file = os.path.join(DATA_DIR, dict_path["path_ff"])

# Loads initial peak positions
peak_positions = pd.read_csv(os.path.join(DATA_DIR, "peak_positions.csv"))
number_of_peaks = len(peak_positions)
print(f"Loaded {number_of_peaks} Bragg peak positions.")

# File is an output of script 02. Contains evolution of center and residuals.
# Only first center position is ectracted.
CRPATH = dict_path["path_center"]
cen_res = np.loadtxt(os.path.join(DATA_DIR, CRPATH), dtype=float, skiprows=1)
center = cen_res[0, 0:2]

# Loads laser background
background_file = os.path.join(DATA_DIR, dict_path["path_bkg"])
LASER_BKG = np.array(
    skued.diffread(os.path.join(DATA_DIR, background_file)), dtype=np.float64
)

# Loads parameters
window_size_intensity = dict_numerics["window_size_intensity"]
window_size_background_peaks = dict_numerics["window_size_background_peaks"]
mask_size_zero_order = dict_numerics["mask_size_zero_order"]
max_distance = dict_numerics["max_distance"]

# Creating dummy image for displaying masks
dummy_image = Image.open(diffraction_images[1])
dummy = np.asarray(dummy_image, dtype="float64")
dummy_bkg = correct_image(
    image_file=diffraction_images[1],
    background_file=background_file,
    flatfield_file=flatfield_file,
)

# %% Loads masks
path_mask = os.path.join(DATA_DIR, dict_path["path_mask"])
mask_total = spio.loadmat(path_mask)["mask_total"]
mask_zero_order = (
    get_mask_image(
        mask_size=dummy.shape,
        center_positions=[center],
        list_of_radii=[mask_size_zero_order],
    )
    * mask_total
)
masked_bragg = (
    get_mask_image(
        mask_size=dummy.shape,
        center_positions=list(
            zip(peak_positions["x_result"], peak_positions["y_result"])
        ),
        list_of_radii=window_size_intensity * np.ones(number_of_peaks),
    )
    * mask_zero_order
)
masked_total_counts = (
    get_mask_image(dummy.shape, [center], [max_distance], True) * mask_zero_order
)
masked_dyn_bg = get_mask_image(
    mask_size=dummy.shape,
    center_positions=list(zip(peak_positions["x_result"], peak_positions["y_result"])),
    list_of_radii=window_size_background_peaks * np.ones(number_of_peaks),
)

# %% Loads peak positions
peak_position_evolution_file = os.path.join(DATA_DIR, dict_path["peak_pos"])

if os.path.exists(peak_position_evolution_file):
    peak_position_evolution = pd.read_csv(peak_position_evolution_file)
    peak_position_evolution = peak_position_evolution.map(ast.literal_eval)
else:
    print("Creating peak position file...")
    peak_position_evolution = get_peak_position_evolution(
        diffraction_image_files=diffraction_images,
        mask_total=mask_total,
        background_file=background_file,
        flatfield_file=flatfield_file,
        peak_positions=peak_positions,
        window_size=dict_numerics["lens_corr_window_size"],
    )
    peak_position_evolution.to_csv(
        os.path.join(DATA_DIR, "peak_position_evolution.csv"), index=False
    )

# %% Path for saving output file
out_file = os.path.join(DATA_DIR, f"normalized_timetraces_{os.path.splitext(os.path.basename(config_file))[0]}.csv")

# %% Display images with masks

cmap = "inferno"
b = 2
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), sharex=True, sharey=True)
vmax = 0.1 * np.nanmax(np.abs(dummy))
plots = [
    (axs[0, 0], dummy, "Original"),
    (axs[0, 1], dummy_bkg, "Laser background removed"),
    (axs[0, 2], dummy_bkg * b, "Background without laser"),
    (axs[1, 0], dummy_bkg * masked_bragg, "Bragg peaks windows"),
    (axs[1, 1], dummy_bkg * masked_total_counts, "Mask for total electron counts"),
    (axs[1, 2], dummy_bkg * masked_dyn_bg, "Mask for dynamical background"),
]

for ax, data, title in plots:
    ax.pcolormesh(data, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

for idp, pp in peak_positions.iterrows():
    axs[1, 0].plot(pp.y, pp.x, marker="x", markersize=4, color="r")
    axs[1, 0].text(pp.y - 10, pp.x - 10, str(idp), color="r")

plt.tight_layout()
output_path = os.path.join(DATA_DIR, "masks.png")
plt.savefig(output_path, format="png")
plt.show()

# %% START ANALYSIS

if os.path.exists(DATA_DIR + "output_data_04.npz"):
    npzfile = np.load(DATA_DIR + "output_data_04.npz")
    total_counts = npzfile["tc"]
    peak_position_evolution = npzfile["pp_evo"]
    intensities_raw = npzfile["int_raw"]
    npzfile.close()
else:
    # Loop over all images collecting peak intensities, total counts etc....
    if partial == 0:
        partial = len(diffraction_images)
    total_counts = np.zeros(len(diffraction_images[0:partial]))
    background_raw = np.zeros(len(diffraction_images[0:partial]))
    intensities_raw = np.zeros((number_of_peaks, len(diffraction_images[0:partial])))

    for idx, diffraction_image in tqdm.tqdm(enumerate(diffraction_images[0:partial])):
        image = correct_image(
            image_file=diffraction_image,
            background_file=background_file,
            flatfield_file=flatfield_file,
        )
        if np.nanmax(np.nanmax(image)) == 65000:
            print("Warning: Image " + str(idx) + " is saturated!")

        total_counts[idx] = np.nansum(image * masked_total_counts)

        for idp, peak in enumerate(peak_position_evolution[f'{idx}']):
            intensities_raw[idp, idx] = sum_peak_pixels(
                image, peak, window_size_intensity
            )

    np.savez(
        os.path.join(DATA_DIR, "output_data_04.npz"),
        pp_evo=peak_position_evolution,
        int_raw=intensities_raw,
        tc=total_counts,
    )

# %% Display total counts and peakpos evolution
plt.figure()
plt.plot(np.arange(len(total_counts)), total_counts)
plt.xlabel("Image number [#]")
plt.ylabel("Total counts [#]")
plt.show()

# %%  Normalize dataset via minimizing correlation of NOE
offset, corr_int = normalize_pearson(
    np.mean(intensities_raw, axis=0), total_counts, tolerance=1e-15, max_steps=10000
)

# Normalize peak intensities
intensities_norm = []
for k in range(0, np.size(intensities_raw, axis=0)):
    intensities_norm.append(np.divide(intensities_raw[k, :], total_counts - offset))
intensities_norm = np.array(intensities_norm, dtype=float)

# Read log file. Get delays, filenames and scans
with open(log_file, "rt") as meta:
    lines = meta.readlines()
del lines[0]

diffraction_images = []
delays = []
scans = []
for line in lines:
    diffraction_images.append(line.split("\t")[0] + ".tif")
    delays.append(float(line.split("\t")[3]))
    scans.append(int(line.split("\t")[2]))
delays = np.array(delays, dtype=float)

no_scans = np.unique(scans)
no_delays = len(np.unique(delays))
number_of_peaks = intensities_norm.shape[0]

# Sort delays and intensities
delay_sort, ind = np.unique(delays, return_inverse=True)
intensities_mean_norm = np.empty((intensities_norm.shape[0], 0), dtype=float)

# Summarize intensities
for ii in range(0, len(delay_sort)):
    temp_ind = np.where(ind == ii)
    intensities_mean_norm = np.hstack(
        (
            intensities_mean_norm,
            np.mean(intensities_norm[:, temp_ind[0]], axis=1).reshape(
                number_of_peaks, 1
            ),
        )
    )

total_int = np.mean(intensities_mean_norm, axis=0)


plt.figure()
plt.plot(delays, intensities_raw[8, :], "o")
plt.show()

plt.figure()
for i in range(0, number_of_peaks):
    plt.plot(
        delay_sort / 1000, intensities_mean_norm[i, :] / intensities_mean_norm[i, 0]
    )
plt.xlabel("Delay $\Delta t$ [ps]")
plt.ylabel("Relative Intensity $\Delta I_r$ [%]")
plt.savefig(
    os.path.join(DATA_DIR, "timetraces.png"),
    format="png",
)
plt.show()

plt.figure()
plt.plot(delay_sort / 1000, total_int / total_int[0])
plt.xlabel("Delay $\Delta t$ [ps]")
plt.ylabel("Mean total intensity $\Delta I_r$ [%]")
plt.savefig(
    os.path.join(DATA_DIR, "total_intensities.png"),
    format="png",
)
plt.show()

np.savez(
    DATA_DIR + "output_data_04_02_norm.npz",
    int_norm=intensities_norm,
    int_mean_norm=intensities_mean_norm,
    tot_int=total_int,
    delay=delay_sort,
    delay_raw=delays,
    offset=offset,
    tc=total_counts,
)
