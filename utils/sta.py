import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.signal import convolve
from skimage import measure


# ------------------------------------------------------------- #
# >> STA computation and analysis (OK)
# ------------------------------------------------------------- #


def compute_3D_sta(
    data: dict,
    checkerboard: np.ndarray,
    nb_frames_per_sequence: int,
    temporal_dimension: int,
    cell_id: int = None,
    min_num_spikes_for_sta: int = 0,
    verbose: bool = True,
):
    """
    Compute the 3D spike triggered average of a cell for the checkerboard stimulus.

    Args:
        data (dict): Dictionary containing the spike counts for the cell {"counted_spikes": 2D array of shape (nb_sequences, nb_frames)}.
        checkerboard (numpy array): 3D array containing the shown checkerboard sequences stacked one after the other with shape (nb_frames, nb_checks, nb_checks).
        nb_frames_per_sequence (int): Number of frames in each sequence of the stimulus.
        temporal_dimension (int): Desired length in time bins (i.e. frames) of the STA.
        cell_id (int, optional): Identifier for the cell being analyzed. Defaults to None.
        min_num_spikes_for_sta (int, optional): Minimum number of spikes required to compute a valid STA. Defaults to 0.

    Returns:
        sta (numpy array): 3D array of shape (temporal_dimension, nb_checks, nb_checks) representing the computed spike-triggered average for the cell. If the total number of spikes is less than or equal to min_num_spikes_for_sta, a zero-like array is returned.
    """
    cell_lab = f"Cell {cell_id}" if cell_id is not None else "The cell"
    nb_sequences = data["counted_spikes"].shape[0]
    sta = np.zeros_like(checkerboard[:temporal_dimension], dtype="float64")
    total_spikes = np.sum(data["counted_spikes"])

    if total_spikes <= min_num_spikes_for_sta:  # added check before computation
        if verbose:
            print(
                f"{cell_lab} has {total_spikes} spikes, less than {min_num_spikes_for_sta} spikes, zero-like sta returned."
            )
        return sta

    for sequence in range(nb_sequences):
        for frame in range(
            temporal_dimension, int(nb_frames_per_sequence / 2)
        ):  # should be made more robust to extact portion is in extract sequence?
            sta_frame_start = (
                sequence * int(nb_frames_per_sequence / 2) + frame - temporal_dimension
            )
            sta_frame_end = sequence * int(nb_frames_per_sequence / 2) + frame
            weight = data["counted_spikes"][sequence, frame]

            sta += weight * checkerboard[sta_frame_start:sta_frame_end, :, :]

    if np.max(np.abs(sta)) > 0:
        sta = sta / total_spikes
        # Bring values between -1 and 1
        sta -= np.median(sta)
        sta /= np.max(np.abs(sta))
    else:
        print(
            f"{cell_lab} null sta upon computation, this should not happen, zero-like sta returned."
        )

    return sta


### (Matias) sta analysis functions


def gaussian2D(
    shape,
    amp,
    x0,
    y0,
    sigma_x,
    sigma_y,
    angle,
):
    if sigma_x == 0:
        sigma_x = 0.001

    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]), int(shape[1]))
    x = np.linspace(0, shape[1], shape[1])
    y = np.linspace(0, shape[0], shape[0])
    X, Y = np.meshgrid(x, y)

    theta = 3.14 * angle / 180
    a = (math.cos(theta) ** 2) / (2 * sigma_x**2) + (math.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(math.sin(2 * theta)) / (4 * sigma_x**2) + (math.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (math.sin(theta) ** 2) / (2 * sigma_x**2) + (math.cos(theta) ** 2) / (
        2 * sigma_y**2
    )

    return amp * np.exp(
        -(
            a * np.power((X - x0), 2)
            + 2 * b * np.multiply((X - x0), (Y - y0))
            + c * np.power((Y - y0), 2)
        )
    )


def gaussian2D_flat(x, amp, x0, y0, rx, ry, rot):
    return gaussian2D(x, amp, x0, y0, rx, ry, rot).flatten()


def reduced_gaussian2D(x, amp, rx, ry, rot):
    # "Reduced" 2D gaussian used for the first fit stage of double_gaussian_fit: the
    # center is held FIXED (passed inside x) and only amplitude/radii/rotation are fit.
    # x packs [ny, nx, x0, y0] — the spatial STA shape and the peak location.
    ny, nx, x0, y0 = x
    return gaussian2D((ny, nx), amp, x0, y0, rx, ry, rot)


def reduced_gaussian2D_flat(x, amp, rx, ry, rot):
    return reduced_gaussian2D(x, amp, rx, ry, rot).flatten()


def double_gaussian_fit(
    spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.unravel_index(np.argmax(np.abs(spatial), axis=None), spatial.shape)
    ydata = spatial.flatten()

    # First fit without center variability
    first_guess = [spatial[center[0], center[1]], 1, 1, 0]
    xdata = [spatial.shape[0], spatial.shape[1], center[1], center[0]]

    ellipse_params_bounds = (
        (-2, 0.1, 0.1, 0),
        (2, spatial.shape[0], spatial.shape[0], 180),
    )

    opt, cov = curve_fit(
        reduced_gaussian2D_flat,
        xdata,
        ydata,
        p0=first_guess,
        bounds=ellipse_params_bounds,
    )

    # Second fit with center variability
    xdata = spatial.shape
    second_guess = [opt[0], center[1], center[0], opt[1], opt[2], opt[3]]

    ellipse_params_bounds = (
        (-2, 0, 0, 0.1, 0.1, 0),
        (
            2,
            spatial.shape[0],
            spatial.shape[0],
            spatial.shape[0],
            spatial.shape[0],
            180,
        ),
    )
    return curve_fit(
        gaussian2D_flat, xdata, ydata, p0=second_guess, bounds=ellipse_params_bounds
    )


def preprocess_fitting_matias(spatial, treshold=0.1):
    sta_spa = spatial.copy()
    sta_treshold = np.max(np.abs(spatial)) * treshold
    sta_spa[np.abs(sta_spa) < sta_treshold] = 0
    return sta_spa


def smooth_sta(sta, alpha, max_time_window=15):
    """
    STA smoothing array by blending each pixel's value with the sum of its 3×3 spatial neighborhood
    controlled by `alpha` (higher alpha = less smoothing) with zero-padding, and locates the coordinates of the peak
    location within the last `max_time_window` time steps.

    Args:
        sta (np.ndarray): 3D array of shape (time, height, width) representing the spike-triggered average.
        alpha (float): Mixing weight between the original pixel value and its neighborhood sum. Values closer to 1 preserve the original signal; values closer to 0 apply stronger smoothing.
        max_time_window (int, default 15): Number of most recent time steps to search when identifying the peak response.

    Returns:
        receptive_field (np.ndarray): Smoothed version of sta, same shape as input.
        tuple(best_t, x, y): Index of the peak absolute response, with best_t adjusted to global time coordinates.
        receptive_field[best] (np.ndarray): Values of the smoothed receptive field at the peak location (a 1D slice or scalar depending on indexing).

    """
    pading_size = 1
    paded_sta = np.pad(sta, pading_size)  # default zero padding
    receptive_field = np.zeros(sta.shape)
    for x in range(sta.shape[1]):
        for y in range(sta.shape[2]):
            receptive_field[:, x, y] = paded_sta[
                1:-1, x + pading_size, y + pading_size
            ] * alpha + (1 - alpha) * paded_sta[
                1:-1,
                x + pading_size - 1 : x + pading_size + 2,
                y + pading_size - 1 : y + pading_size + 2,
            ].sum(axis=(1, 2))

    best = np.unravel_index(
        np.argmax(np.abs(receptive_field[-max_time_window:, :, :])),
        receptive_field.shape,
    )
    best_t = best[0] + max(sta.shape[0] - max_time_window, 0)

    return receptive_field, (best_t, best[1], best[2]), receptive_field[best]


def get_cell_shift(sta):
    sta_3D = sta.copy()
    smooth_sta_1_3D, best_1, max_val1 = smooth_sta(sta_3D, alpha=0.5)
    smooth_sta_2_3D, best_2, max_val2 = smooth_sta(sta_3D, alpha=0.8)

    if abs(max_val1) > abs(max_val2):
        return best_1
    else:
        return best_2


### Standard sta analysis functions (from Tom)


def preprocess_fitting_standard(
    spatial_sta: np.ndarray,
    smoothing_kernel: np.ndarray = None,
) -> np.ndarray:
    """
    Smooth the spatial STA with a 'gaussian' kernel, apply exponential compression and remove low values.

    Args:
        spatial_sta (np.ndarray): 2D array (N, M) representing the spatial STA to be processed.
        smoothing_kernel (np.ndarray, optional): 2D array representing the kernel to be used for smoothing.
        If None, a default kernel resembling a spatial laplacian is used.
    Returns:
        np.ndarray: Processed spatial STA after smoothing, compression, and thresholding (N, M)
    """
    if smoothing_kernel is None:
        # Smooth STA with a Kernel that looks like a spatial laplacian
        smoothing_kernel = np.full((3, 3), 0.2 / 9)
        smoothing_kernel[1, 1] += 0.8

    processed_spatial_sta = convolve(
        spatial_sta, smoothing_kernel, mode="same", method="direct"
    )  # same to keep the same shape, direct to avoid artifacts of fft convolution on small arrays

    # Apply exponential ( == signed power-law) compression and threshold small values
    exponent = 1.25
    noise_threshold = 0.2

    processed_spatial_sta = (
        np.sign(processed_spatial_sta) * np.abs(processed_spatial_sta) ** exponent
    )
    peak = np.max(np.abs(processed_spatial_sta)) * 2
    processed_spatial_sta[
        np.abs(processed_spatial_sta) < peak * noise_threshold**exponent
    ] = 0

    assert processed_spatial_sta.shape == spatial_sta.shape, (
        f"Output shape {processed_spatial_sta.shape} does not match input shape {spatial_sta.shape}"
    )

    return processed_spatial_sta


def get_sta_components(
    sta_3D: np.ndarray, nb_frames=15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """
    Standard function to extract spatial and temporal components from the 3D STA.
    Considering only the last N frames of the 3D STA, find the RF center as the peak standard deviation
    over time of the temporally smoothed 3D STA (spatial_mask), then take the temporal STA as the trace of the 3D STA at
    this spatial location, and finally take the spatial STA as the slice of the 3D STA at the time bin
    corresponding to the absolute max of the temporal STA. Then normalize both components and return them
    together with the spatial mask and the coordinates of the RF center.

        Args:
            sta_3D (np.ndarray): 3D array of shape (time, height, width) representing the spike-triggered average.
            nb_frames (int): Number of most recent time steps to consider when identifying the RF center and extracting components.
        Returns:
            spatial_sta (np.ndarray): 2D array representing the spatial STA extracted as a slice from the 3D STA (unproceessed).
            temporal_sta (np.ndarray): 1D array representing the temporal STA extracted as a trace from the 3D STA (unproceessed).
            spatial_mask (np.ndarray): 2D array representing the standard deviation across time of the temporally smoothed 3D STA, used to identify the RF center.
            cell_delay (int): Time bin corresponding to the spatial STA.
            (cx, cy) (tuple): Spatial unit coordinates corresponding to the temporal STA (respectively (col, row) in the matrix reference).
    """
    # Assume STA signal within the last N frames of the 3d sta and consider only those
    # to help fitting in noisy STAs
    sta3d = sta_3D[-nb_frames:, :, :]

    # Smoothing in time
    kernel_length = 2
    temporal_kernel = (
        np.ones(kernel_length)[:, None, None] / kernel_length
    )  # 1D kernel for temporal smoothing
    temporally_smoothed_sta3d = convolve(
        sta3d, temporal_kernel, mode="valid", method="direct"
    )  # Apply convolution along the temporal dimension (valid to avoid adding extra null time bins through zero-padding, and direct to have consistent convolution computation)

    # Consider the spatial STA as the std across time of the temporally smoothed STA
    spatial_mask = temporally_smoothed_sta3d.std(axis=0)
    # Normalization
    spatial_mask -= np.median(spatial_mask)
    spatial_mask /= np.max(np.abs(spatial_mask))

    # Retrieve temporal STA
    smoothed_spatial_mask = preprocess_fitting_standard(spatial_mask)
    row_max, col_max = np.unravel_index(
        np.argmax(smoothed_spatial_mask), shape=smoothed_spatial_mask.shape
    )
    temporal_sta = sta_3D[:, row_max, col_max]
    # Normalization (to later select the slice for the spatial STA as the outlier on the temporal sta not on the 3d sta)
    temporal_sta -= np.median(temporal_sta)
    temporal_sta /= np.max(np.abs(temporal_sta))

    # Identify the time bin with the maximum response in the temporal STA
    t_max = np.argmax(np.abs(temporal_sta))
    spatial_sta = sta_3D[t_max, :, :]
    spatial_sta -= np.median(spatial_sta)
    spatial_sta /= np.max(np.abs(spatial_sta))

    cell_delay = t_max  # time bin corresponding to the spatial STA
    cx, cy = col_max, row_max  # coordinates of the temporal STA on the spatial STA

    return spatial_sta, temporal_sta, spatial_mask, cell_delay, (cx, cy)


### (Chiara) unifying tom and matias's
def get_temporal_spatial_sta(sta_3D):
    """
    Extracting spatial and temporal STA from the 3D STA by finding the position of the
    maximum in absolute value in the 3D STA and taking the corresponding spatial and temporal traces.
    """
    if np.max(np.abs(sta_3D)) == 0:
        print(
            "Error: empty STA, should be checked upstream - case not handled, returning all None"
        )
        return None, None, None

    # double-attempt  (first stronger than weaker) smoothing + peak location
    best_t, best_row, best_col = get_cell_shift(sta_3D)
    # components extraction
    sta_temporal = sta_3D[:, best_row, best_col]
    sta_spatial = sta_3D[best_t, :, :]
    # max-normalization to have (-1, 1) values
    sta_spatial /= np.max(np.abs(sta_spatial))
    best_x, best_y = best_col, best_row

    return sta_temporal, sta_spatial, (best_t, best_x, best_y)


def _fit_rf_ellipse(fitting_data, error_msg, default_params):
    """Fit the RF ellipse (2D gaussian) on preprocessed spatial data, with a fallback.

    Returns ``(ellipse_params, fitted)``. On failure it prints ``error_msg`` and returns
    ``default_params`` with ``fitted=False``.
    """
    try:
        ellipse_params, _ = double_gaussian_fit(fitting_data)
        return ellipse_params, True
    except Exception:
        print(error_msg)
        return default_params, False


### (Wrap) sta analysis functions wrapped in one function to call easily
def rf_analysis(
    sta_3d: np.ndarray, cell_id: int = None, method: str = "standard"
) -> dict:
    """
    Compute the spatial and temporal STA and fit an ellipse (2d gaussian) on the spatial STA to extract RF parameters.

    Args:
        sta_3d (numpy array): 3D array of shape (nT, nY, nX) containing the 3D STA for a cell.
        cell_id (int, optional): Identifier for the cell being analyzed. Defaults to None.
        method (str, optional): Method to use for STA analysis.

    Returns:
        result (dict): Dictionary containing the following keys:
            - "Spatial": 2D numpy array representing the spatial STA.
            - "Temporal": 1D numpy array representing the temporal STA.
            - "EllipseCoor": List of parameters of the fitted ellipse (amp, x0, y0, sigma_x, sigma_y, rot_angle) in pxs.
            - "Cell_delay": Time bin corresponding to the spatial STA.
            - "FittedEllipse": Boolean indicating whether the ellipse fitting was successful or if default parameters were returned due to an error.
    """
    if cell_id is None:
        cell_id = "cell #"  # for print purposes only
    else:
        cell_id = f"cell {cell_id}"

    error_msg = f"Error in rf analysis of {cell_id} with method {method}: couldn't fit ellipse, default ellipse coord returned"
    def_ellipse_params = [0, 0, 0, 0.001, 0.001, 0]
    default_cell_delay = np.nan
    fitted = False

    if np.max(np.abs(sta_3d)) == 0:
        return {
            "Spatial": np.zeros_like(sta_3d[0]),
            "Spatial_mask": np.zeros_like(sta_3d[0]),
            "Temporal": np.zeros_like(sta_3d[:, 0, 0]),
            "Temporal_STA_coords": (0, 0),
            "EllipseCoor": def_ellipse_params,
            "Cell_delay": default_cell_delay,
            "FittedEllipse": fitted,
        }

    sta3d = sta_3d.copy()

    if method == "matias" or method == "tom":
        temporal_sta, spatial_sta, best = get_temporal_spatial_sta(sta3d)
        cell_delay, cx, cy = best
        cxy = (cx, cy)
        spatial_mask = np.zeros_like(spatial_sta)
        spatial_mask[:] = np.nan
        if method == "matias":
            fitting_data = preprocess_fitting_matias(spatial_sta)
        elif method == "tom":
            fitting_data = preprocess_fitting_standard(spatial_sta)
        else:
            raise ValueError(
                f"You should not arrive here, method should be either 'matias' or 'tom', not {method}"
            )
        ellipse_params, fitted = _fit_rf_ellipse(
            fitting_data, error_msg, def_ellipse_params
        )

    elif method == "standard":
        spatial_sta, temporal_sta, spatial_mask, cell_delay, cxy = get_sta_components(
            sta3d
        )
        smoothed_mask = preprocess_fitting_standard(spatial_mask)
        ellipse_params, fitted = _fit_rf_ellipse(
            smoothed_mask, error_msg, def_ellipse_params
        )

    # elif method == 'guilhem':
    #     time_window_peak_location = 15
    #     fitting_data, _ = gabriel_preprocessing(sta3d, tresholding_factor=1, nb_frames=time_window_peak_location)
    #     try:
    #         ellipse_params, cov = double_gaussian_fit(fitting_data)
    #         temporal_sta = gabriel_temporal_sta(sta3d, ellipse_params)
    #         best_t = np.argmax(np.abs(temporal_sta[-time_window_peak_location:]))
    #         best_t += max(sta3d.shape[0] - time_window_peak_location, 0)
    #         spatial_sta = sta3d[best_t]
    #         cell_delay = best_t
    #         fitted = True
    #     except:
    #         print(error_msg)
    #         plt.imshow(fitting_data)
    #         plt.show(block=False)
    #         spatial_sta = np.zeros_like(sta3d[0])
    #         temporal_sta = np.zeros_like(sta3d[:, 0, 0])
    #         ellipse_params = def_ellipse_params
    #         cell_delay = default_cell_delay
    else:
        raise ValueError(f"Unknown method {method} for rf analysis")

    # wrap results in a dictionary
    result = {
        "Spatial": spatial_sta,
        "Spatial_mask": spatial_mask,
        "Temporal_STA_coords": cxy,
        "Temporal": temporal_sta,
        "EllipseCoor": ellipse_params,
        "Cell_delay": cell_delay,
        "FittedEllipse": fitted,
    }

    # check
    expected_keys = {
        "Spatial",
        "Spatial_mask",
        "Temporal",
        "Temporal_STA_coords",
        "EllipseCoor",
        "Cell_delay",
        "FittedEllipse",
    }
    assert set(result.keys()) == expected_keys, (
        f"Result keys {result.keys()} do not match expected keys {expected_keys}"
    )
    assert isinstance(result["Spatial"], np.ndarray) and result["Spatial"].ndim == 2, (
        f"Spatial STA should be a 2D numpy array, got {type(result['Spatial'])} with ndim {result['Spatial'].ndim}"
    )
    assert (
        isinstance(result["Temporal"], np.ndarray) and result["Temporal"].ndim == 1
    ), (
        f"Temporal STA should be a 1D numpy array, got {type(result['Temporal'])} with ndim {result['Temporal'].ndim}"
    )
    assert (
        isinstance(result["EllipseCoor"], (list, np.ndarray))
        and len(result["EllipseCoor"]) == 6
    ), (
        f"EllipseCoor should be a list or array of 6 parameters, got {type(result['EllipseCoor'])} with length {len(result['EllipseCoor'])}"
    )
    assert isinstance(result["Cell_delay"], (int, np.integer)) or np.isnan(
        result["Cell_delay"]
    ), (
        f"Cell_delay should be a number or NaN, got {type(result['Cell_delay'])} with value {result['Cell_delay']}"
    )

    return result


def plot_sta(
    ax,
    spatial_sta,
    ellipse_params,
    level_factor=0.4,
    color="w",
    alpha=0.8,
    lw=1,
    linestyles="solid",
    cmap="RdBu_r",
    add_center_cross=True,
    marker_size=50,
    marker_symbol="+",
):
    """
    Plot the spatial STA and the fitted ellipse on a given axis,  with colormap centered on 0.
    The ellipse is plotted as a contour at a level defined by `level_factor` times the
    maximum absolute value of the Gaussian fit, and the center of the ellipse can be
    highlighted with a marker if `add_center_cross` is True.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the STA and ellipse
        spatial_sta (numpy.ndarray): 2D array representing the spatial STA to be plotted.
        ellipse_params (list or numpy.ndarray): List or array of 6 parameters (amp, x0, y0, sigma_x, sigma_y, rot_angle) defining the fitted Gaussian ellipse.
        level_factor (float, optional): Factor to multiply the maximum absolute value of the Gaussian fit to determine the contour level for plotting the ellipse.
        color (str, optional): Color for the ellipse contour and center marker.
        alpha (float, optional): Transparency level for the ellipse contour.
        lw (float, optional): Line width for the ellipse contour.
        linestyles (str or list, optional): Line style for the ellipse contour.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap for displaying the spatial STA, centered on 0.
        add_center_cross (bool, optional): Whether to add a marker at the center of the fitted ellipse.
        marker_size (float, optional): Size of the center marker if `add_center_cross` is True.
        marker_symbol (str, optional): Marker symbol for the center marker if `add_center_cross` is True.

    """
    # magnified_ellipse_params=(np.array(ellipse_params)*[gaussian_factor, 1,1,gaussian_factor,gaussian_factor,1])
    gaussian = gaussian2D(spatial_sta.shape, *ellipse_params)
    amp, x0, y0, sigma_x, sigma_y, rot_angle = ellipse_params
    vrange = np.max(np.abs(spatial_sta))
    im = ax.imshow(spatial_sta, vmin=-vrange, vmax=vrange, cmap=cmap)
    if ellipse_params[0] != 0:
        ax.contour(
            np.abs(gaussian),
            levels=[level_factor * np.max(np.abs(gaussian))],
            colors=color,
            linestyles=linestyles,
            alpha=alpha,
            linewidths=lw,
        )
        if add_center_cross:
            ax.scatter(
                x0,
                y0,
                color=color,
                s=marker_size,
                marker=marker_symbol,
                alpha=alpha,
                label="Ellipse center",
            )
    return ax, im


def add_scalebar(
    ax,
    scalebar_size_um,
    pixel_size_um,
    scalebar_left_location=(0.9, 0.9),
    nx=None,
    ny=None,
    scale_bar_color="black",
    scale_bar_width=4,
    label=True,
    fontsize=10,
):
    """Draw a labelled scale bar of ``scalebar_size_um`` micrometres on an image axis.

    The bar is placed in axes fractions (``scalebar_left_location`` = the (x, y) of its
    RIGHT end, both in 0-1) so it stays visible even on a zoomed receptive-field plot,
    and its length is scaled to the current view so it always represents the true
    physical size. When ``label`` is True the size (e.g. "100 µm") is written just below
    the bar. ``nx``/``ny`` are kept for backward compatibility but are no longer used.
    """
    view_px = abs(
        ax.get_xlim()[1] - ax.get_xlim()[0]
    )  # current view width, in STA pixels
    bar_frac = (
        scalebar_size_um / pixel_size_um
    ) / view_px  # bar length as an axes fraction
    x_right, y = scalebar_left_location
    x_left = x_right - bar_frac
    ax.plot(
        [x_left, x_right],
        [y, y],
        transform=ax.transAxes,
        color=scale_bar_color,
        lw=scale_bar_width,
        solid_capstyle="butt",
        clip_on=False,
    )
    if label:
        ax.text(
            (x_left + x_right) / 2,
            y - 0.03,
            f"{scalebar_size_um:.0f} µm",
            transform=ax.transAxes,
            color=scale_bar_color,
            ha="center",
            va="top",
            fontsize=fontsize,
            clip_on=False,
        )
    return


def convert_ellipse_params_to_physical_units(
    ellipse_params: list, sta_pixel_size: float
) -> list:
    """
    Convert ellipse parameters from pixel units to physical units (sta_pixel_size's units).
    Args:
    - ellipse_params: List of ellipse parameters in pixel units [amp, x0_px, y0_px, sigma_x_px, sigma_y_px, rot_angle_deg]
    - sta_pixel_size: Size of one pixel in physical units (e.g., micrometers per pixel)
    Returns:
        List of ellipse parameters in physical units [amp, x0_pu, y0_pu, sigma_x_pu, sigma_y_pu, rot_angle_deg]
    """
    amp, x0_px, y0_px, sigma_x_px, sigma_y_px, rot_angle_deg = ellipse_params

    x0_pu = x0_px * sta_pixel_size
    y0_pu = y0_px * sta_pixel_size
    sigma_x_pu = sigma_x_px * sta_pixel_size
    sigma_y_pu = sigma_y_px * sta_pixel_size

    ellipse_params_pu = [
        amp,  # amplitude (no change)
        x0_pu,
        y0_pu,
        sigma_x_pu,
        sigma_y_pu,
        rot_angle_deg,  # rotation angle (no change)
    ]
    return ellipse_params_pu


def get_temporal_sta_time_vector(
    temporal_sta: np.ndarray, sta_time_bin: float
) -> np.ndarray:
    """
    Generate a time vector for a temporal STA.
    Args:
        - temporal_sta: 1D array representing the temporal STA
        - sta_time_bin: Size of each time bin in temporal units (e.g., seconds)
    Returns:
        - time_vector: 1D array of time delays (-ndt from last bin of temporal_STA) in temporal units
    """
    return np.flip(np.arange(0, -len(temporal_sta) * sta_time_bin, -sta_time_bin))


def get_cell_delay_time(
    cell_delay: int, temporal_sta: np.ndarray, sta_time_bin: float
) -> float:
    """ "
    Convert cell delay in time bins to time in seconds using the temporal STA and the time bin size.
    Args:
        - cell_delay: Cell delay in time bins (integer index of the temporal STA)
        - temporal_sta: 1D array representing the temporal STA
        - sta_time_bin_s: Size of each time bin in temporal units (e.g., seconds)
    Returns:
        - cell_delay_time: Cell delay in temporal units (-ndt) corresponding to the given cell_delay in time bins.
        If cell_delay is None, returns None. If cell_delay is NaN, returns NaN.
    """
    if cell_delay is None:
        return None
    if np.isnan(cell_delay):
        return np.nan
    else:
        tsta_tv = get_temporal_sta_time_vector(temporal_sta, sta_time_bin)
        return tsta_tv[cell_delay]


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the area of a polygon given its vertices using the shoelace formula.

    Args:
        x (numpy.ndarray): 1D array of x-coordinates of the polygon vertices.
        y (numpy.ndarray): 1D array of y-coordinates of the polygon vertices.
    Returns:
        float: Area of the polygon.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ellipse_area(
    ellipse_params: list,
    method: str = "formula",
    level_factor: float = None,
    spatial_sta_shape: tuple = None,
) -> float:
    """
    Calculate the area of an ellipse defined by its parameters.

    Args:
        ellipse_params (list or numpy.ndarray): List or array of 6 parameters (amp, x0, y0, sigma_x, sigma_y, rot_angle) defining the fitted Gaussian ellipse.
        method (str, optional): Method to calculate the area.
            Options are "polygon" for calculating the area of the contour polygon,
            or "formula" (default) for using the mathematical formula of the area of an ellipse (π * sigma_x * sigma_y)
    Returns:
        float: Area of the ellipse calculated according to the method.
    """
    _, _, _, sigma_x, sigma_y, _ = ellipse_params
    if method == "formula":
        return np.pi * sigma_x * sigma_y
    elif method == "polygon":
        assert spatial_sta_shape is not None, (
            "spatial_sta_shape must be provided for polygon method"
        )
        assert level_factor is not None and 0 < level_factor < 1, (
            "level_factor must be provided for polygon method and must be between 0 and 1"
        )
        gaussian = gaussian2D(spatial_sta_shape, *ellipse_params)
        abs_gaussian = np.abs(gaussian)
        contours = measure.find_contours(
            abs_gaussian, level_factor * np.max(abs_gaussian)
        )
        contour = contours[0]  # shape: (N, 2) — note: (row, col) order
        return polygon_area(contour[:, 1], contour[:, 0])  # x=col, y=row
    else:
        raise ValueError(f"Unknown method {method} for ellipse area calculation")


def ellipse_radius(ellipse_params: list, method: str = "circle_approx") -> float:
    """
    Calculate the radius of an ellipse defined by its parameters.

    Args:
        ellipse_params (list or numpy.ndarray): List or array of 6 parameters (amp, x0, y0, sigma_x, sigma_y, rot_angle) defining the fitted Gaussian ellipse.
        method (str, optional): Method to calculate the radius.
            Options are "mean" for computing the mean of the standard deviations in x and y,
            "max" for computing the maximum of the standard deviations in x and y,
            "min" for computing the minimum of the standard deviations in x and y,
            or "circle_approx" (default) for computing the radius of a circle that approximates the ellipse using geometric mean.
    Returns:
        float: Radius of the ellipse calculated according to the method.
    """
    amp, x0, y0, sigma_x, sigma_y, rot_angle = ellipse_params
    if method == "mean":
        # Compute the mean of the standard deviations in x and y
        return np.mean([sigma_x, sigma_y])
    elif method == "max":
        # Compute the maximum of the standard deviations in x and y
        return np.max([sigma_x, sigma_y])
    elif method == "min":
        # Compute the minimum of the standard deviations in x and y
        return np.min([sigma_x, sigma_y])
    elif method == "circle_approx":
        # Compute the radius of a circle that approximates the ellipse
        # using the geometric mean of the standard deviations in x and y
        return np.sqrt(sigma_x * sigma_y)
    else:
        raise ValueError(f"Unknown method {method} for ellipse radius calculation")


def ellipse_diameter(ellipse_params: list, method: str = "circle_approx") -> float:
    # Diameter is simply 2 times the radius
    return 2 * ellipse_radius(ellipse_params, method=method)


def rf_snr(
    spatial_sta: np.ndarray,
    ellipse_params: list,
    method: str = "peak_std",
    level_factor: float = 0.4,
) -> float:
    """
    Calculate the SNR of a spatial STA using a 2D Gaussian defined by ellipse_params.

    Args:
        spatial_sta:   2D NumPy array, the spatial STA
        ellipse_params: parameters passed to gaussian2D (amplitude, x0, y0, sigma_x, sigma_y, theta)
        method:        SNR computation method:
                         - "binary_mask"     : signal = inside gaussian mask, noise = outside
                         - "weighted"        : signal = STA weighted by gaussian, noise = weighted residual
                         - "peak_std"        : signal = peak of STA inside mask, noise = std outside mask
            level_factor:  factor to define the gaussian mask as abs(gaussian) > level_factor * abs(amp)

    Returns:
        SNR: float
    """
    gaussian = gaussian2D(spatial_sta.shape, *ellipse_params)
    amp = ellipse_params[0]
    thr = level_factor * abs(amp)
    mask = abs(gaussian) > thr  # binary mask from the gaussian

    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.imshow(spatial_sta)
    # plt.title("Spatial STA")
    # plt.subplot(1,3,2)
    # plt.imshow(gaussian)
    # plt.title("Gaussian")
    # plt.subplot(1,3,3)
    # plt.imshow(mask)
    # plt.title("Mask")
    # plt.show(block=False)

    if method == "binary_mask":
        # ----------------------------------------------------------------
        # Signal = sum of |STA values| inside the gaussian mask
        # Noise  = sum of |STA values| outside the mask
        # ----------------------------------------------------------------

        signal = np.abs(np.sum(spatial_sta[mask]))
        noise = np.abs(np.sum(spatial_sta[~mask]))

        SNR = signal / noise if noise != 0 else np.inf

    elif method == "weighted":
        # ----------------------------------------------------------------
        # Signal = STA projected onto the gaussian (dot product)
        # Noise  = residual between STA and gaussian-weighted STA
        # ----------------------------------------------------------------
        gaussian_norm = gaussian / gaussian.sum()  # normalize gaussian weights

        signal = np.abs(np.sum(spatial_sta * gaussian_norm))
        residual = spatial_sta - (signal * gaussian_norm)
        noise = np.sqrt(np.mean(residual**2))  # RMS of residual

        SNR = signal / noise if noise != 0 else np.inf

    elif method == "peak_std":
        # ----------------------------------------------------------------
        # Signal = peak absolute value of STA inside the gaussian mask
        # Noise  = standard deviation of STA outside the mask
        # ----------------------------------------------------------------

        if np.sum(mask) == 0:
            # print("Warning: empty mask for SNR calculation, returning SNR=-1")
            return -1

        signal = np.max(np.abs(spatial_sta[mask]))
        noise = np.std(spatial_sta[~mask])

        SNR = signal / noise if noise != 0 else np.inf

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'binary_mask', 'weighted', 'peak_std'."
        )

    return SNR


def check_rf_fit(
    spatial_sta: np.ndarray,
    ellipse_params: list,
    min_amp: float,
    invalid_coords: list,
    min_sigma: float,
    min_rf_area: float,
    min_rf_diameter: float,
    min_rf_snr: float,
    level_factor: float,
    verbose: bool = False,
):
    valid_check = {
        "valid_ellipse_params": False,
        "rf_area": -1,
        "rf_diameter": -1,
        "rf_snr": -1,
        "good_rf": False,
    }
    # check if ellipse_params are valid
    # default params usually are something like [0, 0, 0, 0.001, 0.001, 0]
    amp, x0, y0, sigma_x, sigma_y, rot_angle = ellipse_params
    if (
        abs(amp) <= min_amp  # null amplitude
        or (x0, y0)
        in invalid_coords  # invalid center coordinates (e.g. (0,0) which is often the default)
        or sigma_x <= min_sigma
        or sigma_y <= min_sigma  # very small sigma close to default params
    ):
        return valid_check

    valid_check["valid_ellipse_params"] = True

    # check rf dimensions
    valid_check["rf_area"] = ellipse_area(ellipse_params, method="formula")
    valid_check["rf_diameter"] = ellipse_diameter(
        ellipse_params, method="circle_approx"
    )

    # check SNR
    valid_check["rf_snr"] = rf_snr(
        spatial_sta, ellipse_params, method="peak_std", level_factor=level_factor
    )

    if (
        valid_check["rf_area"] >= min_rf_area
        and valid_check["rf_diameter"] >= min_rf_diameter
        and valid_check["rf_snr"] >= min_rf_snr
    ):
        valid_check["good_rf"] = True

    if verbose:
        print("RF fit check:")
        print(f" - area={valid_check['rf_area']:.2f}")
        print(f" - diameter={valid_check['rf_diameter']:.2f}")
        print(f" - SNR={valid_check['rf_snr']:.2f}")

    return valid_check
