# %%
import sys, os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

# %%
import pandas as pd
import numpy as np

from helpers import fitting
from helpers.tools import center_of_mass, correct_image

center_of_mass_position = pd.DataFrame(columns=["x, y"])
fit_data = pd.DataFrame(columns=["x", "dx", "y", "d"])

# center_of_mass_position.to_csv("data\data-1\CenterOfMass_data.txt")
# fit_data.to_csv("data\data-1\fit_data.txt")

# %%
image_file = "C:/Users/patri/Documents/Topfloor/20190502/meas2/meas2_0015.tif"
background_file = (
    "C:/Users/patri/Documents/Topfloor/20190502/meas2/laser_background/meas1_0039.tif"
)
flatfield_file = "C:/Users/patri/Documents/Topfloor/20180808_flatfield_improved_without_bad_pixel_mask.mat"
peak_positions_filepath = os.path.join(DATA_DIR, "peak_selection_large.txt")
peak_positions = pd.read_csv(peak_positions_filepath, sep="\t")

# %%
image = correct_image(
    image_file=image_file,
    background_file=background_file,
    flatfield_file=flatfield_file,
)
# %%
for index, row in peak_positions.iterrows():
    window_width = int(peak_positions.roi)
    miller_index = peak_positions.miller_index

    xlb = int(peak_positions.x) - window_width
    xub = int(peak_positions.x) + window_width
    ylb = int(peak_positions.y) - window_width
    yub = int(peak_positions.y) + window_width

    x = np.arange(xlb, xub, 1)
    y = np.arange(ylb, yub, 1)

    peak = image[xlb:xub, ylb:yub]

    plt.figure()
    plt.imshow(peak)
    plt.show()

    xpos, ypos = center_of_mass(peak, x, y)
    print(str(xpos) + "  " + str(ypos))

    peak_positions.loc[index,]
#%%
peak_positions.loc[0,"x"]
# %%
paras = []
for j in range(1, len(ps)):
    data = ps[j].split("\t")  # read every line from txt, split columns by tabs

    # You need to create a vector containing all the x positions of the peaks you wish to fit.
    # This could be the output of a previous script that you wrote for the user to pick peaks in an image.

    x0 = float(data[0])  # position x of peak idx.
    y0 = float(data[1])  # position y of peak idy
    # print(x0)
    # print(y0)

    # You also need to create a vector "ROI_BG" containing the widths of the ROI you want to fit. This might be peak dependent.
    ws = int(float(data[3]))
    mil = data[2]
    # print(ws)
    # print(mil)

    # Defines the boundaries of the fits for chosen peak
    xlb = int(x0) - ws
    xub = int(x0) + ws
    ylb = int(y0) - ws
    yub = int(y0) + ws

    x = np.arange(xlb, xub, 1)
    y = np.arange(ylb, yub, 1)

    xgrid, ygrid = np.meshgrid(x, y)

    # Here you will need to insert some code where you load the image you want to fit. The whole diffraction image at this point.
    # It should be an image with FF, BG correction already done, and probably the average image over the different scans.

    peak = IMcor[xlb:xub, ylb:yub]
    # =============================================================================
    #     plt.imshow(peak, cmap=cm.Blues)
    #     plt.show()
    # =============================================================================

    initial_guess = (10000, 0.1, x0, 8, y0, 8, 0.1, 0.1, 200)
    fitbounds = (
        [500, 0, x0 - 10, 2, y0 - 10, 2, -150, -150, 0],
        [100000, 1, x0 + 10, 25, y0 + 10, 25, 150, 150, 2000],
    )

    popt, pcov = opt.curve_fit(
        pseudoVoigt2D_surfBG,
        (xgrid, ygrid),
        peak.ravel(),
        p0=initial_guess,
        check_finite=False,
        bounds=fitbounds,
    )
    sdiv = np.sqrt(np.diag(pcov))

    data_fitted = pseudoVoigt2D_surfBG((xgrid, ygrid), *popt)
    z = data_fitted.reshape(2 * ws, 2 * ws)
    paras.append(popt)

    # Test function here
    plot_comp_fit_exp(x, y, peak, z, save=False, savename="test")

    txt.writelines(
        str(popt[2])
        + "\t"
        + str(sdiv[2])
        + "\t"
        + str(popt[4])
        + "\t"
        + str(sdiv[4])
        + "\n"
    )

txt.close()
