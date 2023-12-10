# %%
import sys, os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

from PIL import Image
import sys
from sklearn.metrics import r2_score
import glob
import numpy as np
import pylab as plt
import pandas as pd
import scipy.optimize as opt
import math as m
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from crystals import Crystal
import helpers.fedutils as utils
from helpers.tools import center_of_mass, correct_image, get_center

# %%
config_file = "config_2020_19_07.cfg"
datapath = "meas2/"
peak_positions_file = "peak_positions.csv"
background_file = "laser_background/meas1_0039.tif"
flatfield_file = "20180808_flatfield_improved_without_bad_pixel_mask.mat"
cif_file = "MoS2_mp-1018809_conventional_standard.cif"

# %%
dict_path, dict_numerics = utils.read_cfg(os.path.join(DATA_DIR, config_file))
electron_energy = dict_numerics["electron_energy"]

peak_positions = pd.read_csv(os.path.join(DATA_DIR, peak_positions_file))
peak_positions[["h", "k", "l"]] = peak_positions.miller_index.str.split(
    " ", expand=True
).astype(float)

crystal = Crystal.from_cif(os.path.join(DATA_DIR, cif_file))
peak_positions["scattering_vector"] = peak_positions[["h", "k", "l"]].apply(
    lambda row: row.iloc[0] * crystal.reciprocal_vectors[0]
    + row.iloc[1] * crystal.reciprocal_vectors[1]
    + row.iloc[2] * crystal.reciprocal_vectors[2],
    axis=1,
)
peak_positions["scattering_vector_length"] = (
    peak_positions["scattering_vector"].apply(lambda row: np.linalg.norm(row)).round(7)
)

center = get_center(peak_positions=peak_positions)


def Nxy(xdata_tuple, a, b, c, d, e):
    # squared function with tilded background
    (x, y) = xdata_tuple
    Nxy = a * (x**2 + y**2) + b + c * x + d * y + e * x * y
    return Nxy.ravel()


def get_fit_show_Nxy(rcom, q_rou, dataCom, center):
    pos = np.array(dataCom, dtype=float)
    x = pos[:, 0]
    y = pos[:, 1]
    # relation A^-1 ~= N*px. One angstrom^-1 should have linear dependence to pixel
    r = np.array(rcom)
    N = np.divide(q_rou, r)
    ws = max(rcom)

    xlb = center[0] - ws
    xub = center[0] + ws
    ylb = center[1] - ws
    yub = center[1] + ws

    xs = np.arange(xlb, xub, 1)
    ys = np.arange(ylb, yub, 1)

    xgrid, ygrid = np.meshgrid(xs, ys)

    # Fit and plot N(x,y)
    initial_guess = (0.1, 0, 0, 0, 0)
    fitbounds = ([0, -10, -10, -10, -10], [10, 10, 10, 10, 10])
    popt, pcov = opt.curve_fit(
        Nxy, (x, y), N.ravel(), p0=initial_guess, check_finite=True, bounds=fitbounds
    )
    sdiv = np.sqrt(np.diag(pcov))

    data_fitted = Nxy((xgrid, ygrid), *popt)
    N_fit = data_fitted.reshape(len(xs), len(ys))
    N_fitd = Nxy((x, y), *popt)

    R2 = r2_score(N, N_fitd)
    print("R^2= ", str(R2))
    print("Paras:", str(popt))
    print("Standard deviations :", str(sdiv), "\n")

    fig = plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor="w", edgecolor="k")
    plt.rcParams["font.size"] = 12
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x, y, N, c=cm.hsv((N - min(N)) / max(N - min(N))), marker="o", edgecolor="k"
    )
    ax.scatter(center[0], center[1], popt[1], c="k")
    sp = ax.plot_surface(xgrid, ygrid, N_fit, cmap=cm.hsv, alpha=0.3)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_zlabel("N(x,y) [$\AA^{-1}$/px]")  # ,linespacing=15
    ax.zaxis.set_label_coords(100, 100, 1)
    ax.dist = 8
    ax.tick_params(axis="z", labelrotation=45)
    ax.ticklabel_format(style="sci", axis="z", scilimits=(-2, 2))
    plt.show()
    return N


def res_func(xdata_tuple, A, f, g, h, a, b):
    (x, y) = xdata_tuple
    res = (
        A
        * np.sin(f * (x * np.cos(g) - y * np.sin(g)) - a)
        * np.sin(h * (x * np.sin(g) + y * np.cos(g)) - b)
    )
    return res


def give_fit_paras(popt, pcov, data, fit, print_results):
    sdiv = np.sqrt(np.diag(pcov)) 
    res = data - fit
    mean_res = np.mean(abs(res))

    r2 = r2_score(data, fit)

    if print_results == 1:
        print(f"Mean residuals: {mean_res}")
        print(f"R^2= {r2}")
        print(f"Paras: {popt}")
        print(f"Standard deviations :{sdiv}")

    return sdiv, res, mean_res, r2


def fit_qi(qi, x, y, xgrid, ygrid, show_plot):
    initial_guess = (0, 0, 0, 0, 0)
    fitbounds = ([-1, -1, -1, -1, -10], [1, 1, 1, 1, 10])
    popt, pcov = opt.curve_fit(
        utils.qi_func,
        (x, y),
        qi.ravel(),
        p0=initial_guess,
        check_finite=True,
        bounds=fitbounds,
    )
    qi_fit = utils.qi_func((x, y), *popt)
    sdiv, rqi, mean_rqi, R2 = give_fit_paras(popt, pcov, qi, qi_fit, 0)

    qi_mesh = utils.qi_func((xgrid, ygrid), *popt)
    qi_mesh = qi_mesh.reshape(len(y_con), len(x_con))

    if show_plot == 1:
        # plot qi(x,y)
        fig = plt.figure(
            num=None, figsize=(12, 8), dpi=150, facecolor="w", edgecolor="k"
        )
        plt.rcParams["font.size"] = 12
        ax = fig.add_subplot(111, projection="3d")
        colmap = cm.ScalarMappable(cmap=cm.hsv)
        colmap.set_array(qi) 
        ax.scatter(
            x,
            y,
            qi,
            c=cm.hsv((qi - min(qi)) / max(qi - min(qi))),
            marker="o",
            edgecolor="k",
        )
        ax.scatter(0, 0, 0, c="k")
        ax.plot_surface(xgrid, ygrid, qi_mesh, cmap=cm.hsv, alpha=0.3)
        fig.colorbar(colmap)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_zlabel("$q_i(x,y)$ [$\AA^{-1}$]") 
        ax.zaxis.set_label_coords(100, 100, 1)
        ax.dist = 10
        ax.tick_params(axis="z", labelrotation=45)
        ax.ticklabel_format(style="sci", axis="z", scilimits=(-2, 2))
        plt.show()

        # plot residuals
        fig = plt.figure(
            num=None, figsize=(12, 8), dpi=150, facecolor="w", edgecolor="k"
        )
        plt.rcParams["font.size"] = 12
        ax = fig.add_subplot(111, projection="3d")
        ax.bar3d(x, y, 0, 10, 10, rqi)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_zlabel("Residuals [$\AA^{-1}$]")  # ,linespacing=15
        ax.set_zlim([min(rqi), max(rqi)])
        ax.dist = 10
        ax.ticklabel_format(style="sci", axis="z", scilimits=(-1, 1))
        ax.tick_params(axis="z", labelrotation=45)
        plt.show()

    return popt, sdiv, rqi, mean_rqi, R2


# %%
# Get distances of peak pos. This part calculates a conversion factor of one
# diffraction pattern to estimate how one pixel distance is in reciprocal space.
# Path of COM positons of one diffraction pattern calculated by script 01

peak_positions["center_distance"] = peak_positions.apply(
    lambda row: m.sqrt(
        (center[0] - row["x_result"]) ** 2 + (center[1] - row["y_result"]) ** 2
    ),
    axis=1,
)

# %% Get anf fit conversion factor N(x,y) [A^-1/px]

N = get_fit_show_Nxy(
    peak_positions["center_distance"].to_numpy(),
    peak_positions["scattering_vector_length"].to_numpy(),
    peak_positions[["x_result", "y_result"]].to_numpy(),
    center,
)
# ------------------------------------------------------------------------------
# %% Get tif files from data set dir
files = glob.glob(os.path.join(DATA_DIR, datapath) + "*.tif")
number_tif = len(files)

# %%
# Write peak and center positions
posFile = os.path.join(DATA_DIR, "pos_com_data.txt")

# Write fit parameters in text
qxFile = os.path.join(DATA_DIR, "qx_fit_data.txt")
dataqx = open(qxFile, "w")
dataqx.writelines("a" + "\t" + "b" + "\t" + "c" + "\t" + "d" + "\t" + "e" + "\n")

qyFile = os.path.join(DATA_DIR, "qy_fit_data.txt")
dataqy = open(qyFile, "w")
dataqy.writelines("a" + "\t" + "b" + "\t" + "c" + "\t" + "d" + "\t" + "e" + "\n")

dataFile = os.path.join(DATA_DIR, "cen_res_data.txt")
data = open(dataFile, "w")
data.writelines(
    "center-x" + "\t" + "center-y" + "\t" + "qx-residual" + "\t" + "qy-residual" + "\n"
)

dataps = peak_positions[["x", "y", "miller_index", "roi"]].to_numpy()
qvec = np.vstack(peak_positions["scattering_vector"].to_numpy()).reshape(3,-1)
q_rou = peak_positions["scattering_vector_length"].to_numpy()

for k in range(0, number_tif):
    # Apply corrections to image
    print(f"File: {k} of {number_tif}")
    IMcor = correct_image(
        image_file=files[k],
        background_file=os.path.join(DATA_DIR, background_file),
        flatfield_file=os.path.join(DATA_DIR, flatfield_file),
    )

    positions = peak_positions.copy()
    pos = np.empty((0, 2))
    for jj in range(0, len(dataps)):
        x0 = float(dataps[jj][0])  # position x of peak idx.
        y0 = float(dataps[jj][1])  # position y of peak idy

        ws = int(float(dataps[jj][3]))

        # Defines the boundaries of the fits for chosen peak
        xlb = int(x0) - ws
        xub = int(x0) + ws
        ylb = int(y0) - ws
        yub = int(y0) + ws

        x = np.arange(xlb, xub, 1)
        y = np.arange(ylb, yub, 1)

        peak = IMcor[xlb:xub, ylb:yub]

        # Apply center of mass method
        xpos, ypos = center_of_mass(peak, x, y)

        pos = np.vstack((pos, [xpos, ypos]))

    positions[["x_result","y_result"]] = pos
    # Get center (position of zero order peak)
    center = get_center(positions)

    # Create grid
    x = pos[:, 0] - center[0]
    y = pos[:, 1] - center[1]
    xlb = np.round(min(x))
    xub = np.round(max(x))
    ylb = np.round(min(y))
    yub = np.round(max(y))

    x_con = np.arange(xlb, xub, 10)
    y_con = np.arange(ylb, yub, 10)

    xgrid, ygrid = np.meshgrid(x_con, y_con)

    px, sdx, rqx, mean_rqx, R2x = fit_qi(qvec[0, :], x, y, xgrid, ygrid, 0)
    py, sdy, rqy, mean_rqy, R2y = fit_qi(qvec[1, :], x, y, xgrid, ygrid, 0)

    data.writelines(
        str(center[0])
        + "\t"
        + str(center[1])
        + "\t"
        + str(mean_rqx)
        + "\t"
        + str(mean_rqy)
        + "\t"
        + str(R2x)
        + "\t"
        + str(R2y)
        + "\n"
    )
    dataqx.writelines(
        str(px[0])
        + "\t"
        + str(px[1])
        + "\t"
        + str(px[2])
        + "\t"
        + str(px[3])
        + "\t"
        + str(px[4])
        + "\n"
    )
    dataqy.writelines(
        str(py[0])
        + "\t"
        + str(py[1])
        + "\t"
        + str(py[2])
        + "\t"
        + str(py[3])
        + "\t"
        + str(py[4])
        + "\n"
    )
data.close()
dataqx.close()
dataqy.close()


# plot center evolution
plt.rcParams["font.size"] = 12
# Plot of center of mass data
plt.figure(num=None, figsize=(5, 4), dpi=100, facecolor="w", edgecolor="k")
plt.plot(np.arange(1, number_tif + 1, 1), cen_data[:, 0] - cen_data[0, 0], "ro", ms=2)
plt.xlabel("Image number [#]", fontsize=12)
plt.ylabel("Center shift [px]", fontsize=12)
plt.dist = 12
plt.savefig(
    os.path.join(DATA_DIR, "center_shift.png"),
    format="png",
)
plt.show()
