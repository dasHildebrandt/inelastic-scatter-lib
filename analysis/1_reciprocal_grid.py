""" 
Returns adjusted peak positions of Bragg peaks in a diffraction pattern
"""
# %%
import sys, os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from helpers import fitting
from helpers.tools import center_of_mass, correct_image

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

# %%
image_file = os.path.join(DATA_DIR, "meas2/meas2_0015.tif")
background_file = os.path.join(DATA_DIR, "laser_background/meas1_0039.tif")
flatfield_file = os.path.join(
    DATA_DIR, "20180808_flatfield_improved_without_bad_pixel_mask.mat"
)
peak_positions_filepath = os.path.join(DATA_DIR, "peak_selection_large.txt")
peak_positions = pd.read_csv(peak_positions_filepath, sep="\t")

# %%
image = correct_image(
    image_file=image_file,
    background_file=background_file,
    flatfield_file=flatfield_file,
)

# %%
for index, peak in peak_positions.iterrows():
    window_width = int(peak.roi)
    miller_index = peak.miller_index

    xlb = int(peak.x) - window_width
    xub = int(peak.x) + window_width
    ylb = int(peak.y) - window_width
    yub = int(peak.y) + window_width

    x = np.arange(xlb, xub, 1)
    y = np.arange(ylb, yub, 1)

    peak_image = image[xlb:xub, ylb:yub]

    plt.figure()
    plt.imshow(peak_image)
    plt.show()

    xpos, ypos = center_of_mass(peak_image, x, y)

    peak_positions.loc[index, "x_result"] = xpos
    peak_positions.loc[index, "y_result"] = ypos

peak_positions.to_csv(os.path.join(DATA_DIR, "peak_positions.csv"), index=False)
