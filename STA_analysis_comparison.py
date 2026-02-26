from utils import gaussian2D, gaussian2D_flat
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve

def get_sta_components(sta_3D, nb_frames=15, kernel_length=2, tresholding_factor=2):
    # Take last frames of the sta to help fitting in noisy STAs
    sta3d = sta_3D[-nb_frames:, :, :]

    # Smoothing in time
    temporal_kernel = np.ones(kernel_length)[:, None, None] / kernel_length # 1D kernel for temporal smoothing
    temporally_smoothed_sta3d = convolve(sta3d, temporal_kernel, mode="valid", method="direct") # Apply convolution along the temporal dimension (valid to avoid adding extra time bins through padding, and direct to have consistent convolution computation)

    # Consider the spatial STA as the variance across time of the temporally smoothed STA
    spatial_sta = temporally_smoothed_sta3d.var(axis=0)
    # Normalization
    spatial_sta -= np.median(spatial_sta)
    spatial_sta /= np.max(np.abs(spatial_sta))

    # Retrieve temporal STA
    smoothed_spatial_sta = utils.preprocess_fitting_standard(spatial_sta)
    x_max, y_max = np.unravel_index(np.argmax(smoothed_spatial_sta), shape=spatial_sta.shape)
    temporal_sta = sta_3D[:, x_max, y_max]
    
    return spatial_sta, temporal_sta
    # gaussian_kernel = gaussian2D(
    #     shape, gaussian_params[0], x_max, y_max, *gaussian_params[3:]
    # )
    # gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    # # Weighted temporal trace
    # return np.mean(gaussian_kernel[None, :, :] * sta_3D, (1, 2))

    # # Thresholding
    # tresholding_factor = 2
    # k_gauss = 1.5  # 1.5 mad ~ 1 std for gaussian noise
    # mad = np.median(np.abs(data - np.median(data)))
    # data[data < tresholding_factor * k_gauss * mad] = 0

    return spatial_sta, temporal_sta

# def gabriel_temporal_sta(sta_3D, gaussian_params):
#     shape = (sta_3D.shape[1], sta_3D.shape[2])
#     smoothing_kernel = gaussian2D(shape, *gaussian_params)
#     smoothing_kernel /= np.sum(smoothing_kernel)
#
#     # Find max in space
#     smoothed_sta = math.convolve(sta_3D.var(0), smoothing_kernel, mode="nearest")
#     x_max, y_max = np.unravel_index(np.argmax(smoothed_sta), shape=shape)
#     # Gaussian weighting kernel
#
#     gaussian_kernel = gaussian2D(
#         shape, gaussian_params[0], x_max, y_max, *gaussian_params[3:]
#     )
#     gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
#     # Weighted temporal trace
#     return np.mean(gaussian_kernel[None, :, :] * sta_3D, (1, 2))

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

# def analyse_sta_gabriel(sta, cell_id):
#     sta_3D = sta.copy()
#
#     try:
#         fitting_data, spatial_sta = gabriel_preprocessing(sta_3D)
#         ellipse_params, cov = fit_gaussian(fitting_data)
#         temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
#
#     except Exception as e:
#         print(f"Error Could not fit ellipse {cell_id}")
#         fitting_data, spatial_sta = gabriel_preprocessing(sta_3D)
#         temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
#         plt.imshow(fitting_data)
#         plt.show(block=False)
#         return {
#             "Spatial": spatial_sta,
#             "Temporal": temporal_sta,
#             "EllipseCoor": [0, 0, 0, 0.001, 0.001, 0],
#             "Cell_delay": np.nan,
#         }
#         print(e)
#
#     return {
#         "Spatial": spatial_sta,
#         "Temporal": temporal_sta,
#         "EllipseCoor": ellipse_params,
#         "Cell_delay": np.nan,
#     }

# ----------------------------------------------------- #

import os

root = r'C:\Users\cboscarino\Documents\GitHub\Standard_analysis_pipeline\data\20251219_PulsingGratings_PupilSize\Analysis\Checkerboard_Analysis_rec_0'
data_filename = "sta_data_analysed.pkl"
data_fp = os.path.join(root, data_filename)
sta_data = utils.load_obj(data_fp)

cell_ids = [1, 2, 9, 10]

for cell_id in cell_ids:
    print(f"Processing cell {cell_id}...")
    sta = sta_data[cell_id]["sta_3D"]

    # Gabriel extraction of sta components
    sta_3D = sta.copy()

    spatial_sta, temporal_sta = get_sta_components(sta_3D)
    smoothed_spatial_sta = utils.preprocess_fitting_standard(spatial_sta)
    ellipse_params, cov = utils.double_gaussian_fit(smoothed_spatial_sta)