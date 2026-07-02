import numpy as np
import pickle
import os
import gc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
from scipy.signal import convolve
from skimage import measure
import itertools
import time
from collections import defaultdict
import matplotlib.path as mpltPath
from types import ModuleType

import params

from .sta import *  # gaussian2D and STA helpers

###########################################################
###########          Registration Holo          ###########
###########################################################


# Functions


def find_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


# ------------------------------------------------------------------------ #
# >>> OLD FUNCTIONS
# ------------------------------------------------------------------------ #

####  Gabriel's STA analysis  ####


def gabriel_preprocessing(sta_3D, nb_frames=15, kernel_lenght=2, tresholding_factor=2):
    data = sta_3D[-nb_frames:, :, :]

    # smoothing along time
    kernel = np.ones(kernel_lenght)[:, None, None] / kernel_lenght
    data = math.convolve(data, kernel, mode="nearest")

    ## Take variance
    data = data.var(0)
    data -= np.median(data)
    data /= np.max(np.abs(data))
    spatial_sta = data.copy()

    ## Thresholding
    tresholding_factor = 2
    k_gauss = 1.5  # 1.5 mad ~ 1 std for gaussian noise
    mad = np.median(np.abs(data - np.median(data)))
    data[data < tresholding_factor * k_gauss * mad] = 0

    return data, spatial_sta


def gabriel_temporal_sta(sta_3D, gaussian_params):
    shape = (sta_3D.shape[1], sta_3D.shape[2])
    smoothing_kernel = gaussian2D(shape, *gaussian_params)
    smoothing_kernel /= np.sum(smoothing_kernel)

    # Find max in space
    smoothed_sta = math.convolve(sta_3D.var(0), smoothing_kernel, mode="nearest")
    x_max, y_max = np.unravel_index(np.argmax(smoothed_sta), shape=shape)
    # Gaussian weighting kernel

    gaussian_kernel = gaussian2D(
        shape, gaussian_params[0], x_max, y_max, *gaussian_params[3:]
    )
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    # Weighted temporal trace
    return np.mean(gaussian_kernel[None, :, :] * sta_3D, (1, 2))


def fit_gaussian(sta_spatial):
    center = np.unravel_index(np.argmax(sta_spatial, axis=None), sta_spatial.shape)
    guess = [np.max(sta_spatial), center[1], center[0], 1, 1, 0]

    xdata = sta_spatial.shape
    ydata = sta_spatial.flatten()

    ellispe_params_bounds = (
        (-2, 0, 0, 0.1, 0.1, 0),
        (
            2,
            sta_spatial.shape[0],
            sta_spatial.shape[0],
            sta_spatial.shape[0],
            sta_spatial.shape[0],
            180,
        ),
    )

    return curve_fit(
        gaussian2D_flat, xdata, ydata, p0=guess, bounds=ellispe_params_bounds
    )


####  Matias's STA analysis ####


def matias_temporal_spatial_sta(sta_3D):
    if np.max(np.abs(sta_3D)) == 0:
        # print(f"Cell {cell_id} : Could not find sta")
        return "Error detected : 3D sta empty", "Error detected : 3D sta empty"

    best_t, best_row, best_col = get_cell_shift(sta_3D)
    sta_temporal = sta_3D[:, best_row, best_col]
    sta_spatial = sta_3D[best_t, :, :]
    sta_spatial /= np.max(np.abs(sta_spatial))
    best_x, best_y = best_col, best_row

    return sta_temporal, sta_spatial, (best_t, best_x, best_y)


### Guilhem's STA analysis ### (mixed between both)


### Tom's STA analysis ### (new fitting of ellipse with new denoising and smoothing of STAs)


# New display with max and min equal and new coulor
def plot_sta_tom(ax, spatial_sta, ellipse_params, level_factor=0.4):
    gaussian = gaussian2D(spatial_sta.shape, *ellipse_params)

    vmax = np.max([np.amax(spatial_sta), -np.amin(spatial_sta)])
    ax.imshow(spatial_sta, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    if ellipse_params[0] != 0:
        ax.contour(
            np.abs(gaussian),
            levels=[level_factor * np.max(np.abs(gaussian))],
            colors="y",
            linestyles="solid",
            alpha=0.4,
            lw=5,
        )
    return ax


### Analysis to quantify the presence of STAs
def check_presence_STA(
    sta, ellipse_coor, nb_of_pixels_by_check, tresh_snr=2.75, level_factor=0.2
):  # Used to check the presence of STAs
    pxl_size_dmd = params.pxl_size_dmd

    gaussian = gaussian2D(sta.shape, *ellipse_coor)

    x0 = ellipse_coor[1]
    y0 = ellipse_coor[2]

    # See if the STA is in the center
    xshape = sta.shape[0]
    yshape = sta.shape[1]
    if x0 > 0.8 * xshape or x0 < 0.2 * xshape or y0 > 0.8 * yshape or y0 < 0.2 * yshape:
        return [0.1, 0.1]

    # See if the STA has a fitted ellipse
    if ellipse_coor[0] != 0:
        plt.figure()
        cs = plt.contour(
            np.abs(gaussian), levels=[level_factor * np.max(np.abs(gaussian))]
        )
        contour = cs.allsegs
        plt.close()

        # Verify that the diameter of the ellipse is neither too big nor too small
        area = polygon_area(contour[0][0][:, 0], contour[0][0][:, 1])
        diameter = 2 * np.sqrt(area / np.pi) * nb_of_pixels_by_check * pxl_size_dmd

        if diameter < 100 or diameter > 500:
            return [0.3, diameter]

        # Verify that the SNR is superior to the threshold of the SNR
        if SNR_test(sta, contour) < tresh_snr:
            return [0.4, SNR_test(sta, contour)]

    else:
        return [0.2, 0.2]

    return [1, SNR_test(sta, contour)]


def SNR_test(sta, contour):  # Calculate the SNR of cells
    path = mpltPath.Path(contour[0][0])
    points = []

    for x_id in range(sta.shape[0]):
        for y_id in range(sta.shape[1]):
            points.append([x_id, y_id])

    inside = path.contains_points(points)

    noise = []
    signal = []
    for ins_id in range(len(inside)):
        if inside[ins_id] is False:
            noise.append(sta[points[ins_id][1], points[ins_id][0]])
        else:
            signal.append(sta[points[ins_id][1], points[ins_id][0]])
            noise.append(0)

    len(signal)
    len(noise)

    noise_compression = []

    for nb_comp in range(sta.shape[0]):
        noise_compression.append(np.mean(noise[nb_comp * 40 : (nb_comp + 1) * 40]))

    noise = np.sum(np.abs(noise_compression))
    signal = np.abs(np.sum(signal))

    SNR = signal / noise

    return SNR
