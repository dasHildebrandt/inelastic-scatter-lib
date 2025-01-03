import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image


def correct_image(
    image_file: str,
    background_file: str,
    flatfield_file: str,
) -> np.ndarray:
    """Returns image array, optional background and flatfield corrected.

    Args:
        image_file (str): path to image file
        background_file (str): path to background file
        flatfield_file (str): path to flatfield file

    Returns:
        np.ndarray: _description_
    """
    assert flatfield_file.lower().endswith(
        ".mat"
    ), "The file does not have a .mat extension"
    assert background_file.lower().endswith(".tif")

    image = Image.open(image_file)
    corrected_image = np.asarray(image, dtype="float64")
    if background_file:
        background_file = Image.open(background_file)
        background_image = np.asarray(background_file, dtype="float64")
        correct_image = corrected_image - background_image
    if flatfield_file:
        FFmat = sio.loadmat(flatfield_file)
        FFimg = FFmat["FF"]  # changes if FF file chnges
        correct_image = np.multiply(FFimg, corrected_image)
    return corrected_image


def center_of_mass(matrix: np.ndarray, x: float, y: float) -> tuple:
    """Returns a center of mass coordinates.

    Args:
        matrix (np.ndarray): matrix with mass or intensity values
        x (float): x coordinate vectors (around peak)
        y (float): y coordinate vectors (around peak)

    Returns:
        (x,y) (tuple): coordinates of the center of mass
    """

    size = matrix.shape
    if size[0] != len(x) and size[1] != len(y):
        print("Error: width is not matching matrix size!")
    else:
        Y, X = np.meshgrid(y, x)
        mass = np.nansum(np.nansum(matrix))
        xpos = np.nansum(np.nansum(matrix * X)) / mass
        ypos = np.nansum(np.nansum(matrix * Y)) / mass
    return (xpos, ypos)


def arrayIndex(M, ind):
    # Array and indices are numpy arrays. If you want to adress elements Matlab
    # like with an array. New array is an array of elements with indices.
    new_array = []
    for j in range(0, len(ind)):
        new_array = np.append(new_array, M[ind[j]])
    return new_array


def get_center(peak_positions: pd.DataFrame) -> tuple:
    """Returns the center beam of the diffraction pattern (transmitted beam), calculated
    from the mean positions of the lowest order Bragg peak pairs.

    Args:
        peak_positions (pd.DataFrame): Dataframe containing the Bragg peak properties

    Returns:
        tuple: position of the center peak
    """
    lowest_order_peaks = peak_positions[
        peak_positions["scattering_vector_length"]
        == peak_positions["scattering_vector_length"].min()
    ]
    miller_indices = lowest_order_peaks[["h", "k", "l"]]
    miller_indices_inverse = lowest_order_peaks[["h", "k", "l"]] * -1
    peak_pair_indices = [
        (i, j)
        for i, row1 in miller_indices.iterrows()
        for j, row2 in miller_indices_inverse.iterrows()
        if row1.equals(row2)
    ]
    unique_peak_pairs = set(tuple(sorted(pair)) for pair in peak_pair_indices)
    unique_peak_pairs = list(unique_peak_pairs)

    if not unique_peak_pairs:
        raise ValueError("No peak pairs found for center peak position calculation.")
    else:
        print(
            f"Center calculated from the mean of {len(unique_peak_pairs)} peak pair position."
        )

    indices = [item for tup in unique_peak_pairs for item in tup]
    lowest_order_peaks = peak_positions.iloc[indices]
    return (
        lowest_order_peaks["x_result"].mean(),
        lowest_order_peaks["y_result"].mean(),
    )
