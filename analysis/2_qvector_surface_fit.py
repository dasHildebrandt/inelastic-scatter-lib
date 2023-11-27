# %%
import sys, os
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.getenv("ROOT_DIR")
DATA_DIR = os.getenv("DATA_DIR")
sys.path.append(ROOT_DIR)

# %%
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
from helpers.tools import center_of_mass, correct_image

# %%
config_file = "config_2020_19_07.cfg"
datapath = "meas2/"
peak_positions_file = "peak_positions.csv"
background_file = "laser_background/meas1_0039.tif"
flatfield_file = "20180808_flatfield_improved_without_bad_pixel_mask.mat"
cif_file = "MoS2_mp-1018809_conventional_standard.cif"

#%%
dict_path, dict_numerics = utils.read_cfg(os.path.join(DATA_DIR, config_file))
electron_energy = dict_numerics["electron_energy"]

peak_positions = pd.read_csv(os.path.join(DATA_DIR, peak_positions_file))
peak_positions[["h","k","l"]]=peak_positions.miller_index.str.split(' ', expand=True).astype(float)

#%%

def get_center(dataCom, dataps, q_rou):
    # Determine center by getting mean of miller pair positions nearest to zero
    # order peak. Those peaks are less effected by non-rotational symmetric
    # squeezing and therefore most suited for center calculations.
    mil_sel = []
    xcom = []
    ycom = []

    ind = np.where(q_rou == min(q_rou))  # lowest Bragg peaks
    # select only first peaks
    tempcom = []
    tempps = []
    for j in range(0, len(ind[0])):
        tempcom.append(dataCom[ind[0][j]])
        tempps.append(dataps[ind[0][j]])
    dataComs = tempcom
    datapss = tempps

    for j in range(0, len(datapss)):
        mil_sel.append(datapss[j][2].replace("-", ""))  # erase '-' fromindices
    mil_sel = np.array(mil_sel)  # all miller indices
    umil_sel = np.array(list(set(mil_sel)))  # unique millerindices

    #
    for j in range(0, len(umil_sel)):
        ind = np.where(mil_sel == umil_sel[j])
        if len(ind[0]) == 2 or len(ind[0]) == 4:
            for jj in range(0, len(ind[0])):
                # print(str(mil_sel[ind[0][jj]]))

                xcom.append(float(dataComs[ind[0][jj]][0]))
                ycom.append(float(dataComs[ind[0][jj]][1]))

    # center for center of mass
    xcom = np.array(xcom)  # convert to array format
    ycom = np.array(ycom)
    cen = np.array([np.mean(xcom, axis=0), np.mean(ycom, axis=0)])
    return cen


def Nxy(xdata_tuple, a, b, c, d, e):
    # squared function with tilded background
    (x, y) = xdata_tuple
    Nxy = a * (x**2 + y**2) + b + c * x + d * y + e * x * y
    return Nxy.ravel()


def get_fit_show_Nxy(rcom, q_rou, dataCom, cen):
    pos = np.array(dataCom, dtype=float)
    x = pos[:, 0]
    y = pos[:, 1]
    # relation A^-1 ~= N*px. One angstrom^-1 should have linear dependence to pixel
    r = np.array(rcom)
    N = np.divide(q_rou, r)
    ws = max(rcom)

    xlb = cen[0] - ws
    xub = cen[0] + ws
    ylb = cen[1] - ws
    yub = cen[1] + ws

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

    # plot N(x,y)
    fig = plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor="w", edgecolor="k")
    plt.rcParams["font.size"] = 12
    ax = fig.add_subplot(111, projection="3d")
    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(N)  # cm.hsv(arg) arg is between 0-1. Normalize z to [0,1].
    ax.scatter(
        x, y, N, c=cm.hsv((N - min(N)) / max(N - min(N))), marker="o", edgecolor="k"
    )
    ax.scatter(cen[0], cen[1], popt[1], c="k")
    sp = ax.plot_surface(xgrid, ygrid, N_fit, cmap=cm.hsv, alpha=0.3)
    fig.colorbar(colmap)
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
    sdiv = np.sqrt(np.diag(pcov))  # Standard diviation
    res = data - fit  # residuals
    mean_res = np.mean(abs(res))

    r2 = r2_score(data, fit)

    if print_results == 1:
        print("Mean residuals: ", str(mean_res))
        print("R^2= ", str(r2))
        print("Paras:", str(popt))
        print("Standard deviations :", str(sdiv), "\n")

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
        colmap.set_array(qi)  # cm.hsv(arg) arg is between 0-1. Normalize z to [0,1].
        # cm.hsv(arg) arg is between 0-1. Normalize z to [0,1].
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
        ax.set_zlabel("$q_i(x,y)$ [$\AA^{-1}$]")  # ,linespacing=15
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

#%%

crystal = Crystal.from_cif(os.path.join(DATA_DIR, cif_file))
kVec = np.array(crystal.reciprocal_vectors, float)

#%%
q, qvec = utils.get_qVectors(kVec, peak_positions[["h","k","l"]])
q_rou = np.around(q, decimals=7)


# Estimate error of reciprocal space projection on screen
me = 9.109383e-31
ec = 1.602176634e-19
h = 6.62607015e-34
ind_high_peak = np.where(q_rou == np.max(q_rou))[0][0]
dbwl = h / np.sqrt(2 * me * ec * electron_energy * 1e3)
k0 = 2 * m.pi / dbwl
max_hkl = hkl[ind_high_peak, :]
d_hkl = (
    2
    * m.pi
    / np.linalg.norm(
        max_hkl[0] * kVec[0, :] + max_hkl[1] * kVec[1, :] + max_hkl[2] * kVec[2, :]
    )
)
theta = np.arcsin(dbwl / (2 * d_hkl * 1e-10))
proj_q = q[ind_high_peak]
G = 2 * k0 * 1e-10 * np.sin(theta)
if np.round(G - proj_q, 15) < 0:
    raise ValueError("q and kVec are not from the same crystal!")

real_q = k0 * np.tan(2 * theta) * 1e-10
proj_err = real_q - proj_q
print("Projection error of highest peak: " + str(proj_err) + " A^-1")


# ------------------------------------------------------------------------------
# Get distances of peak pos. This part calculates a conversion factor of one
# diffraction pattern to estimate how one pixel distance is in reciprocal space.
# Path of COM positons of one diffraction pattern calculated by script 01
com_data_path = (
    "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/centerOfMass_data.txt"
)

dataCom = np.loadtxt(com_data_path, dtype=float, skiprows=1)
cen = get_center(dataCom, dataps, q_rou)
rcom = []
for j in range(0, len(dataps)):
    rcom.append(
        m.sqrt(
            (cen[0] - float(dataCom[j][0])) ** 2 + (cen[1] - float(dataCom[j][1])) ** 2
        )
    )

# Get anf fit conversion factor N(x,y) [A^-1/px]

N = get_fit_show_Nxy(rcom, q_rou, dataCom, cen)
# ------------------------------------------------------------------------------

# Get tif files from data set dir
files = glob.glob(datapath + "*.tif")  # get file list
number_tif = len(files)

# Write peak and center positions
posFile = "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/pos_com_data.txt"

# Write fit parameters in text
qxFile = "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/qx_fit_data.txt"
dataqx = open(qxFile, "w")
dataqx.writelines("a" + "\t" + "b" + "\t" + "c" + "\t" + "d" + "\t" + "e" + "\n")

qyFile = "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/qy_fit_data.txt"
dataqy = open(qyFile, "w")
dataqy.writelines("a" + "\t" + "b" + "\t" + "c" + "\t" + "d" + "\t" + "e" + "\n")

dataFile = "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/cen_res_data.txt"
data = open(dataFile, "w")
data.writelines(
    "center-x" + "\t" + "center-y" + "\t" + "qx-residual" + "\t" + "qy-residual" + "\n"
)

for k in range(0, number_tif):
    # Apply corrections to image
    print("File:", str(k), " of ", str(number_tif))
    img = Image.open(files[k])
    IM = np.asarray(img, dtype="float64")
    if BGfile:
        BG = Image.open(BGfile)
        BGimg = np.asarray(BG, dtype="float64")
        IMcor = IM - BGimg
    if FFfile:
        FFmat = sio.loadmat(FFfile)
        FFimg = FFmat["FF"]  # changes if FF file chnges
        IMcor = np.multiply(FFimg, IMcor)

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
        xpos, ypos = com(peak, x, y)

        pos = np.vstack((pos, [xpos, ypos]))

    # Get center (position of zero order peak)
    cen = get_center(pos, dataps, q_rou)

    # Create grid
    x = pos[:, 0] - cen[0]
    y = pos[:, 1] - cen[1]
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
        str(cen[0])
        + "\t"
        + str(cen[1])
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
    "//nap33.rz-berlin.mpg.de/hildebrandt/Masterarbeit/MoS2/center_shift.png",
    format="png",
)
plt.show()

