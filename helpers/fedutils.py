import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
import configparser
import tqdm
import skued
from scipy.stats import pearsonr
import math as m

from helpers.tools import correct_image, center_of_mass


def read_cfg(path_cfg):
    config = configparser.ConfigParser()
    config.read(path_cfg)

    assert "PATH" in config, "Could not find PATH in the config file."
    assert "PARAMETERS" in config, "Could not find PARAMETERS in the config file."

    dict_path = {}
    for key in config["PATH"]:
        dict_path.update({key: config["PATH"][key]})

    dict_numerics = {}
    for key in config["PARAMETERS"]:
        try:
            dict_numerics.update({key: int(config["PARAMETERS"][key])})
        except:
            dict_numerics.update(config.get("PARAMETERS", "peak_index_update_center"))

    return dict_path, dict_numerics


def get_mask_image(
    mask_size: tuple,
    center_positions: list[tuple],
    list_of_radii: list[float],
    mask_inverse: bool = False,
) -> np.ndarray:
    """
    Returns a mask in image size as a numpy array containing zeros and ones values. The mask should be
    multiplied with the diffraction pattern.

    Args:
        mask_size (tuple): Size of the diffraction image to be masked.
        center_positions (list[float]): Positions of the transmitted diffraction beams.
        list_of_radii (list[float]): Radius of
        mask_inverse (bool, optional): Inverse . Defaults to False.

    Returns:
        mask (ndarray): Mask image cotaining zeros and ones for multiplication with diffraction pattern.
    """
    mask = np.ones(mask_size, dtype=float)
    assert len(center_positions) == len(list_of_radii)
    for idx, center in enumerate(center_positions):
        xc = center[1]
        yc = center[0]
        radius = list_of_radii[idx]
        xx, yy = np.meshgrid(np.arange(0, mask.shape[0]), np.arange(0, mask.shape[1]))
        rr = np.empty_like(xx)
        rr = np.hypot(xx - xc, yy - yc)
        if mask_inverse is False:
            mask[rr <= radius] = np.nan
        elif mask_inverse is True:
            mask[rr >= radius] = np.nan
    return mask


def refine_peak_positions(
    peak_positions: pd.DataFrame, image: np.ndarray, window_size: int
) -> pd.DataFrame:
    peak_positions_image = []
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
        xpos, ypos = center_of_mass(peak_image, x, y)
        peak_positions_image.append((xpos, ypos))
    return peak_positions_image


def centeredDistanceMatrix(n):
    # make sure n is odd
    x, y = np.meshgrid(range(n), range(n))
    return np.sqrt((x - (n / 2) + 1) ** 2 + (y - (n / 2) + 1) ** 2)


def centeredDistanceMatrix_centered(n, xc, yc):
    # make sure n is odd
    x, y = np.meshgrid(range(n), range(n))
    return np.sqrt((x - (n / 2 + xc) + 1) ** 2 + (y - (n / 2 + yc) + 1) ** 2)


# Taken from Laurent
def azimuthal_average(image, center, mask=None, angular_bounds=None, trim=True):
    """
    This function returns an azimuthally-averaged pattern computed from an image,
    e.g. polycrystalline diffraction.
    Parameters
    ----------
    image : array_like, shape (M, N)
        Array or image.
    center : array_like, shape (2,)
        coordinates of the center (in pixels).
    mask : `~numpy.ndarray` or None, optional
        Evaluates to True on valid elements of array.
    angular_bounds : 2-tuple or None, optional
        If not None, the angles between first and second elements of `angular_bounds`
        (inclusively) will be used for the average. Angle bounds are specified in degrees.
        0 degrees is defined as the positive x-axis. Angle bounds outside [0, 360) are mapped back
        to [0, 360).
    trim : bool, optional
        If True, leading and trailing zeros (possible due to the usage of masks) are trimmed.
    Returns
    -------
    radius : `~numpy.ndarray`, ndim 1
        Radius of the average [px]. ``radius`` might not start at zero, depending on the ``trim`` parameter.
    average : `~numpy.ndarray`, ndim 1
        Angular-average of the array.
    """
    if mask is None:
        mask = np.ones_like(image, dtype=np.bool)

    xc, yc = center

    # Create meshgrid and compute radial positions of the data
    # The radial positions are rounded to the nearest integer
    # TODO: interpolation? or is that too slow?
    Y, X = np.indices(image.shape)
    R = np.hypot(X - xc, Y - yc)
    Rint = np.rint(R).astype(np.int)

    if angular_bounds:
        mi, ma = _angle_bounds(angular_bounds)
        angles = (
            np.rad2deg(np.arctan2(Y - yc, X - xc)) + 180
        )  # arctan2 is defined on [-pi, pi] but we want [0, pi]
        in_bounds = np.logical_and(mi <= angles, angles <= ma)
    else:
        in_bounds = np.ones_like(image, dtype=np.bool)

    valid = mask[in_bounds]
    image = image[in_bounds]
    Rint = Rint[in_bounds]

    px_bin = np.bincount(Rint, weights=valid * image)
    r_bin = np.bincount(Rint, weights=valid)
    radius = np.arange(0, r_bin.size)

    # Make sure r_bin is never 0 since it it used for division anyway
    np.maximum(r_bin, 1, out=r_bin)

    # We ignore the leading and trailing zeroes, which could be due to
    first, last = 0, -1
    if trim:
        first, last = _trim_bounds(px_bin)

    return radius[first:last], px_bin[first:last] / r_bin[first:last]


def _trim_bounds(arr):
    """Returns the bounds which would be used in numpy.trim_zeros"""
    first = 0
    for i in arr:
        if i != 0.0:
            break
        else:
            first = first + 1
    last = len(arr)
    for i in arr[::-1]:
        if i != 0.0:
            break
        else:
            last = last - 1
    return first, last


def rings_to_average(d, y, n):
    x = np.arange(n)
    f = interp1d(x, y, fill_value="extrapolate")
    return f(d.flat).reshape(d.shape)


def remove_bgk(image, laser_background, flatfield):
    # Locations where background is larger than the image shouldn't be negative, but zero
    image = np.array(image, dtype=np.float64)
    image[laser_background > image] = 0
    mask = laser_background <= image
    image = mask * (image - laser_background) * flatfield
    return image


def sum_peak_pixels(image, peak, window_size):
    lbx = int(peak[1]) - window_size
    ubx = int(peak[1]) + window_size
    lby = int(peak[0]) - window_size
    uby = int(peak[0]) + window_size
    im_p = image[lby:uby, lbx:ubx]
    return np.nansum(im_p)


def get_peak_position_evolution(
    diffraction_image_files: list[str],
    mask_total: np.ndarray,
    background_file: str,
    flatfield_file: str,
    peak_positions: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    peak_position_evolution = pd.DataFrame()
    for k, image_file in tqdm.tqdm(enumerate(diffraction_image_files)):
        image = correct_image(
            image_file=image_file,
            background_file=background_file,
            flatfield_file=flatfield_file,
        )
        image = image * mask_total
        peak_position_evolution[k] = refine_peak_positions(
            peak_positions=peak_positions, image=image, window_size=window_size
        )
    return peak_position_evolution


def normalize_pearson(signal_intensity, total_counts, tolerance=1e-15, max_steps=10000):
    corr_int = 1
    offset = 0
    counter = 0
    while corr_int > 0:
        offset = 10**counter
        signal_intensity_normalized = signal_intensity / (total_counts - offset)
        # normalize with this offset value
        # calculate correlation with total counts
        corr_int, pp = pearsonr(signal_intensity_normalized, total_counts)
        counter = counter + 1
    offset_min = 10 ** (counter - 2)
    offset_max = 10 ** (counter - 1)
    print(corr_int)
    print(offset)
    counter = 1

    while abs(corr_int) > tolerance and counter < max_steps:
        # take new offset value in the middle
        offset = (offset_max - offset_min) / 2 + offset_min
        # normalize with this offset value
        signal_intensity_normalized = signal_intensity / (total_counts - offset)
        # calculate correlation with total counts
        corr_int, pp = pearsonr(signal_intensity_normalized, total_counts)
        # adjust the offset interval according to the sign of corr_int (the
        # right offset lies at corr=0, so between positive and negative
        # correlation)
        if corr_int < 0:
            offset_max = offset
        else:
            offset_min = offset
        counter = counter + 1

    return offset, corr_int


def reshape(signal_raw, no_delays, scans, no_scans, exclude):
    signal_chunked = []
    for scan in no_scans:
        if scans.count(scan) == no_delays:
            if scan not in exclude:
                if scan % 2 == 0:
                    signal_chunked.append(
                        signal_raw[scan * no_delays : scan * no_delays + no_delays, :]
                    )
                elif scan % 2 == 1:
                    # Stage goes back and forth in our experiment
                    invert = signal_raw[
                        scan * no_delays : scan * no_delays + no_delays, :
                    ][::-1]
                    signal_chunked.append(invert)
        else:
            print("excluding scans " + str(scan))

    signal_chunked = np.array(signal_chunked)
    return signal_chunked


def normalize_pret0(signal_chunked, delays, t0_cutoff=-1000):
    idx_neg = np.where(delays < t0_cutoff)[0]
    signal_chunked_nor = []
    for chunk in signal_chunked:
        chunk = chunk / np.mean(chunk[idx_neg], axis=0)
        signal_chunked_nor.append(chunk)
    return signal_chunked_nor


def qi_func(xdata_tuple, a, b, c, d, e):
    # squared function with tilded background
    (x, y) = xdata_tuple
    x = x * np.cos(e) - y * np.sin(e)
    y = x * np.sin(e) + y * np.cos(e)
    # qi_xy =  a*(x**2+y**2)+b*x+c*y+d*x*y
    qi_xy = np.sign(x) * a * x**2 + b * x + c * (x**2 + y**2) + d * y * x
    return qi_xy


def get_rotsym_BZs(qvec, rot_sym, tol):
    # Get rotational symmetric BZs. Those are BZs/Bragg peaks which have the same center
    # distance and scattering vector axis angle.
    q_rou = np.round(
        np.linalg.norm(qvec, axis=1), tol
    )  # radius of q vectors with complete BZ
    if qvec.shape[1] == 3:
        qvec = qvec.transpose()

    ref_vec = qvec[0:2, 0]
    ang = []
    sym_BZs = []

    # Calculate angle to reference vector
    for k in range(0, qvec.shape[1]):
        ang.append(clockwise_angle(ref_vec, qvec[0:2, k]) + m.pi)

    ang = np.round(ang, tol - 1)
    rot_ang = 2 * m.pi / rot_sym  # one symmetry rotation
    ang_mod = np.round(np.mod(ang, rot_ang), tol - 2)  # modulo

    # BZs are rotational symmetric if modulo and radius are the same.
    q_ang = np.vstack((q_rou, ang_mod))  # creating modulo-radius vector
    q_ang_uni = np.unique(q_ang, axis=1)
    for j in range(0, q_ang_uni.shape[1]):
        temp_ind = []
        for k in range(0, q_ang.shape[1]):
            if q_ang[0, k] == q_ang_uni[0, j] and q_ang[1, k] == q_ang_uni[1, j]:
                temp_ind.append(k)
        sym_BZs.append(temp_ind)
    return sym_BZs


def rot_avg(qvec, rot_sym, sym_BZs):
    # Transpose vector array to shape (3 x n)
    if qvec.shape[1] == 3:
        qvec = qvec.transpose()
    qvec = np.round(qvec, 5)

    # Get indices of rot symmetric BZs
    rot_ang = 360 / rot_sym
    orderlist = []
    for i in range(0, len(sym_BZs)):
        temp_qvecs = qvec[:, sym_BZs[i]]

        # Check self consistency if all vectors have same length
        sc = np.round(np.linalg.norm(temp_qvecs, axis=0), 4)
        if not sc[0] == np.sum(sc) / len(sc):
            print("Error: Selected vectors dont have the same length!")
        else:
            # Continue with analysis
            order = [0]
            for j in range(1, rot_sym):
                rot_vec = vec_rot_3D_z(j * rot_ang, temp_qvecs[:, 0])

                temp_diff = np.empty(temp_qvecs.shape)
                for m in range(0, temp_qvecs.shape[1]):
                    temp_diff[:, m] = temp_qvecs[:, m] - rot_vec
                temp_diff = np.round(np.linalg.norm(temp_diff, axis=0), 1)
                match = np.where(temp_diff == 0)[0]

                if match:
                    order.append(match[0])
                else:
                    order.append("NaN")
        orderlist.append(order)
    return orderlist


def vec_rot_3D_z(ang_d, vec):
    ang = np.radians(ang_d)  # Conversion to radians
    r = np.array(
        [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]],
        dtype=float,
    )
    rot_vec = np.matmul(r, vec)
    return rot_vec


def clockwise_angle(a, b):
    # Gives clockwise angle in 0-360 deg of two 2D vectors.
    dot = np.dot(a, b)  # dot product between [x1, y1] and [x2, y2]
    det = a[0] * b[1] - b[0] * a[1]  # determinant
    angle = m.atan2(det, dot)
    return angle


def reduce_BZ_to_rotsym_part(bravTyp, kpoints, kVec):
    # Bravais typ = [1,2, ... 5]
    # Kpoints handed out by Monkhorst-Pack ASE module

    # Monoclinic
    if bravTyp == 1:
        raise ValueError("Not implemented!")

    # Orthorombic
    if bravTyp == 2:
        kpoints = kpoints

    # Rhombus
    if bravTyp == 3:
        raise ValueError("Not implemented!")

    # Hexagonal Bravais lattice
    if bravTyp == 4:
        minkp = np.min(kpoints, axis=0)
        kpoints = kpoints - np.matlib.repmat(minkp, kpoints.shape[0], 1)

        if not np.sum(kpoints[:, 2]) < 1e-15:
            raise ValueError(
                "Z-component of inplane vectors is not zero. Check lattice vector order."
            )

        if not kVec[0, 1] < 1e-15 and kVec[0, 2] < 1e-15:
            raise ValueError("First vector must be paralle to x-axis.")

        # Cut kpoints matching to 1 BZ conditions
        theta = -np.degrees(clockwise_angle(kVec[0, :], np.array([1, 0, 0])))
        temp_kpoints = np.empty(kpoints.shape)
        for j in range(0, temp_kpoints.shape[0]):
            temp_kpoints[j, :] = vec_rot_3D_z(theta, kpoints[j, :])
        temp_b1 = vec_rot_3D_z(theta, kVec[0, :])
        kpoints = kpoints[
            np.where(np.round(temp_kpoints[:, 0], 3) <= temp_b1[0] * 0.5)[0], :
        ]

        theta = -np.degrees(clockwise_angle(np.array([0, 1, 0]), kVec[1, :]))
        temp_kpoints = np.empty(kpoints.shape)
        for j in range(0, temp_kpoints.shape[0]):
            temp_kpoints[j, :] = vec_rot_3D_z(theta, kpoints[j, :])
        temp_b2 = vec_rot_3D_z(theta, kVec[1, :])
        kpoints = kpoints[np.where(temp_kpoints[:, 1] <= temp_b2[1] * 0.5)[0], :]

        kpoints = kpoints[np.where(kpoints[:, 0] > 1e-15)[0], :]

        full_kpoints = kpoints
        rot_sym = 60
        for k in range(1, 6):
            rot_angle = k * rot_sym
            temp = np.zeros(kpoints.shape)
            for j in range(0, kpoints.shape[0]):
                temp[j, :] = vec_rot_3D_z(rot_angle, kpoints[j, :])
            full_kpoints = np.vstack((full_kpoints, temp))
        kpoints = np.vstack((np.zeros((1, 3)), full_kpoints))

        print(
            "Due to transformation of the Monkhorst-Pack grid to a rotational symmetric grid, the number of peaks changes."
        )

    # tetragonal
    if bravTyp == 5:
        kpoints = kpoints

    return kpoints
