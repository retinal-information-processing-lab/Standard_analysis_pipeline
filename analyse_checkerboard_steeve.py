"""Module of functions for the checkerboard analysis

contact: laquitainesteeve@gmail.com
"""

# import packages
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec
from types import ModuleType

# import custom packages
import utils


# ------------------------------------------------------------------------------------------------------------------- #
# CHECKERBOARD DATA LOADING AND PREPARATION
# ------------------------------------------------------------------------------------------------------------------- #

def prompt_user_for_checkerboard_params() -> tuple[int, int, int, int]:
    """
    Prompt user for stimulus parameters.

    Returns:
        Tuple of (stimulus_frequency, nb_checks_x, nb_checks_y)
    """
    stimulus_frequency = int(input("Select stimulus frequency (Hz, usually in stimulus filename): "))
    nb_checks_x = int(
        input("Select number of checks on x (sq, usually in stimulus filename): ")
    )
    nb_checks_y = int(
        input("Select number of checks on y (sq, usually in stimulus filename): ")
    )
    nb_pixels_per_check = int(input("Select number of pixels per check (px, usually in stimulus filename): "))

    return stimulus_frequency, nb_checks_x, nb_checks_y, nb_pixels_per_check


def get_all_inputs_for_checkerboard_analysis(
    params: ModuleType,
) -> tuple[int, str, float, int, int, int, str]:
    """
    Get all input parameters for checkerboard experiment analysis.

    Interactively prompts user for recording selection, stimulus frequency,
    and checkerboard dimensions. Creates analysis output directory.

    Args:
        params: Dictionary with experiment parameters including:
            - recording_names: List of available recording names
            - output_directory: Base directory for outputs

    Returns:
        Tuple containing:
            - recording_number: Selected recording index
            - stimulus_frequency: Stimulus frequency in Hz
            - nb_checks_x: Number of checkerboard squares in x dimension
            - nb_checks_y: Number of checkerboard squares in y dimension
            - nb_pixels_per_check: Number of pixels per checkerboard square
            - check_directory: Path to analysis output directory
    """
    recording_number, recording_name = utils.prompt_user_for_recording(
        params, "checkerboard"
    )
    stimulus_frequency, nb_checks_x, nb_checks_y, nb_pixels_per_check = prompt_user_for_checkerboard_params()
    check_directory = utils.create_analysis_directory(
        params, recording_number, "Checkerboard"
    )

    return (
        recording_number,
        recording_name,
        stimulus_frequency,
        nb_checks_x,
        nb_checks_y,
        nb_pixels_per_check,
        check_directory,
    )


def calculate_checkerboard_experiment_stats(
        triggers: np.ndarray,
        params: ModuleType,
        stimulus_frequency: float
) -> tuple[int, int]:
    """
    Calculate and display experiment statistics.

    Args:
        triggers: Array of trigger times
        params: Module from params.py containing experiment parameters including 'nb_frames_by_sequence'
        stimulus_frequency: Stimulus frequency in Hz

    Returns:
        Tuple of (nb_repeats, duration_sequence)
    """
    nb_repeats = int(len(triggers) / params.nb_frames_by_sequence)
    duration_sequence = int(params.nb_frames_by_sequence / stimulus_frequency)

    print("\nCheckerboard Stats :")
    print(f"\t- {int(triggers[-1] / 60)} min total duration")
    print(f"\t- {len(triggers)} triggers")
    print(f"\t- {nb_repeats} complete sequences")
    print(f"\t- {duration_sequence} seconds per sequence\n")

    return nb_repeats, duration_sequence


def load_or_create_checkerboard_stimulus(
    nb_repeats: int,
    nb_checks_x: int,
    nb_checks_y: int,
    check_directory: str,
    params: ModuleType,
) -> np.ndarray:
    """
    Load existing stimulus array or create new one.

    Args:
        nb_repeats: Number of stimulus repetitions
        nb_checks_x: Number of checks in x dimension
        nb_checks_y: Number of checks in y dimension
        check_directory: Directory for stimulus file
        params: Module from params.py containing experiment parameters including 'nb_frames_by_sequence' and 'binary_source_path'

    Returns:
        Checkerboard stimulus array
    """
    nb_frames = int(nb_repeats * int(params.nb_frames_by_sequence / 2))
    stimulus_path = os.path.normpath(
        os.path.join(
            check_directory,
            f"checkerboard_{nb_checks_x}x{nb_checks_y}checks_{nb_frames}frames.npy",
        )
    )

    if os.path.isfile(stimulus_path):
        print(f"Stimulus file exists. Loaded from:\t {stimulus_path}")
        checkerboard = np.load(stimulus_path)
    else:
        print("Reconstructing the stimulus...")
        checkerboard = utils.checkerboard_from_binary(
            nb_frames,
            nb_checks_x,
            nb_checks_y,
            checkerboard_file=stimulus_path,
            binary_source_path=params.binary_source_path,
        )

    return checkerboard


def load_checkerboard_data(
    params: ModuleType,
    check_directory: str,
    checkerboard_name: str,
    nb_checks_x: int,
    nb_checks_y: int,
    stimulus_frequency: float
) -> tuple:
    """
    Load and process all checkerboard experiment data.

    Loads triggers, spikes, and stimulus data. Calculates experiment statistics.

    Args:
        params: Module from params.py containing experiment parameters including 'triggers_directory' and 'exp'
        check_directory: Directory for analysis outputs
        checkerboard_name: Name of the checkerboard stimulus
        nb_checks_x: Number of checkerboard squares in x
        nb_checks_y: Number of checkerboard squares in y
        stimulus_frequency: Stimulus frequency in Hz

    Returns:
        Tuple containing:
            - checkerboard_spikes: Dict mapping cell IDs to spike times
            - triggers: Array of trigger times
            - nb_repeats: Number of complete stimulus sequences
            - cells_id: List of cell IDs
            - checkerboard: Stimulus array
    """
    print(params.triggers_directory)
    triggers_path = os.path.normpath(
        os.path.join(
            params.triggers_directory,
            f"{params.exp}_{checkerboard_name}_triggers.pkl",
        )
    )
    print(f"Loading triggers from:\t {triggers_path}")
    stim_onsets = utils.load_stim_onset_from_triggers_path(
        triggers_path, params, verbose=True
    )
    cells_id, checkerboard_spikes = utils.load_spike_times(params, checkerboard_name)
    nb_repeats, _ = calculate_checkerboard_experiment_stats(
        stim_onsets, params, stimulus_frequency
    )
    checkerboard = load_or_create_checkerboard_stimulus(
        nb_repeats, nb_checks_x, nb_checks_y, check_directory, params
    )
    print(f"Checkerboard stimulus shape: {checkerboard.shape}")

    print(f"\nTotal : {len(checkerboard_spikes.keys())} neurons loaded\nCell ids: {[int(x) for x in cells_id]}\n")

    return checkerboard_spikes, stim_onsets, nb_repeats, cells_id, checkerboard


# ------------------------------------------------------------------------------------------------------------------- #
# RESPONSE EXTRACTION
# ------------------------------------------------------------------------------------------------------------------- #

def extract_all_cell_responses_to_repeated_sequences(
        checkerboard_spikes: dict, 
        triggers: np.ndarray, 
        nb_repeats: int, 
        stimulus_frequency: float,
        cells_id: list,
        nb_frames_per_sequence: int,
        sequence_portion: tuple = (0.5, 1)
        ) -> dict:
    
    """
    Extract responses to repeated stimulus sequences for all cells.

    Args:
        checkerboard_spikes: Dict mapping cell IDs to spike times {cell_id: np.array of spike times}
        triggers: Array of trigger times (n_triggers,)
        nb_repeats: Number of complete stimulus sequences
        stimulus_frequency: Stimulus frequency in Hz
        cells_id: List of cell IDs to process
        nb_frames_per_sequence: Number of frames in each stimulus sequence  
        sequence_portion: Tuple specifying which portion of sequence to use (e.g., (0.5, 1) for second half)

    Returns:
        Dict containing extracted responses for each cell, including:
            - spike_trains: List of spike trains for each repetition
            - repeated_sequences_times: List of start and end times for each repeated sequence

            output_data = {
                cell_id (np.uint): {
                    'spike_times': np.array of shape (n_spikes,)
                    'repeated_sequences_times': list of length nb_repeats
                    'spike_trains': list of length nb_repeats
                    'counted_spikes': np.array of shape (nb_repeats, n_bins)
                    'psth': np.array of shape (n_bins,)
                },
    """
    
    # initialise output
    output_data = {}

    # loop over the spikes recorded during the checkerboard experiment
    # get the responses to the repeated sequence
    for cell_id, spike_times in tqdm(checkerboard_spikes.items(), desc="Extracting responses to repeated sequence for each cell"):
        output_data[cell_id] = utils.extract_from_sequence(
            spike_times, 
            triggers, 
            nb_repeats, 
            stim_frequency=stimulus_frequency,
            sequence_portion=sequence_portion,
            nb_frames_per_sequence=nb_frames_per_sequence,
        )

    # check data
    assert set(output_data.keys()) == set(cells_id), "Error in extracting data: Cell IDs in output data do not match expected cell IDs"
    for cell_id in cells_id:
        assert 'spike_times' in output_data[cell_id] and output_data[cell_id]['spike_times'].ndim==1, f"Error in extracting data: 'spike_times' key missing or None for cell ID {cell_id}"
        assert 'repeated_sequences_times' in output_data[cell_id] and isinstance(output_data[cell_id]['repeated_sequences_times'], list) and len(output_data[cell_id]['repeated_sequences_times']) == nb_repeats, f"Error in extracting data: 'repeated_sequences_times' key missing or not a list for cell ID {cell_id}"
        assert 'spike_trains' in output_data[cell_id] and isinstance(output_data[cell_id]['spike_trains'], list) and len(output_data[cell_id]['spike_trains']) == nb_repeats, f"Error in extracting data: 'spike_trains' key missing or not a list for cell ID {cell_id}"
        assert 'counted_spikes' in output_data[cell_id] and output_data[cell_id]['counted_spikes'].ndim==2 and output_data[cell_id]['counted_spikes'].shape[0] == nb_repeats, f"Error in extracting data: 'counted_spikes' key missing or not a 2D array for cell ID {cell_id}"
        nbins = output_data[cell_id]['counted_spikes'].shape[1]
        assert 'psth' in output_data[cell_id] and output_data[cell_id]['psth'].ndim==1 and output_data[cell_id]['psth'].shape[0] == nbins, f"Error in extracting data: 'psth' key missing or not a 1D array for cell ID {cell_id}"
    return output_data


def plot_all_rasters(
        rep_seq_data: dict,
        cells_id: list, 
        fontsize: int = 35, 
        show_labels: bool = False,
        plotting: bool = True):
    """ 
    Plot rasters for all cells in a grid (might take a few seconds).

    Args:
        rep_seq_data: Dict containing extracted responses for each cell (from extract_all_cell_responses_to_repeated_sequences)
        cells_id: List of cell IDs to plot
        plotting: Boolean indicating whether to plot or not
        show_labels: Boolean indicating whether to show axis labels or not
        fontsize: Font size for titles and labels

    Returns:
            None (plots are displayed if plotting is True)
    """
    if plotting:
        size = int(math.sqrt(len(cells_id))) + 1

        # setup subplots
        fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(50, 50))
        for i in tqdm(range(size**2), desc="Plotting rasters for all cells"):
            ax = axs[i // size, i % size]
            if i < len(cells_id):
                ax.eventplot(rep_seq_data[cells_id[i]]["spike_trains"])
                ax.set_title(f"C{cells_id[i]}", fontsize=fontsize)
                if show_labels: 
                    ax.set_xlabel("Time (s)", fontsize=fontsize)
                if show_labels: 
                    ax.set_ylabel("n repetition", fontsize=fontsize)
            else:
                ax.set_visible(False)

        # format and close
        plt.tight_layout()
        plt.show(block=False)
        plt.close("all")
    return None


def plot_and_save_single_cell_rasters(
    rep_seq_data: dict, 
    check_directory: str, 
    folder_name: str = "Rasters_figs",
    cell_ids: list = None, 
    title: str = "Response to repeated sequence",
    fontsize: int = 14,
    save_figures: bool = True,
    show_figures: bool = False,
    save_format: str = "png",
):
    """
    Generate single cell raster plots (one figure per cell showing raster+psth) 
    and save them in a new folder: "Rasters_figs" in the check_directory.

    Args:
        rep_seq_data: Dict containing extracted responses for each cell (from extract_all_cell_responses_to_repeated_sequences)
        check_directory: path to the directory where to generate the subfolder and save the figures
        folder_name: Name of the subfolder to save the figures in
        cell_ids: List of cell IDs to plot (if None, plots all cells)
        title: Title to add to each figure
        fontsize: Font size for titles and labels
        save_figures: Boolean indicating whether to save the figures as files
        show_figures: Boolean indicating whether to display the figures
        save_format: String indicating the format to save the figures in (e.g., "png", "jpg", "svg")
        
    Returns:
        None
    """

    # figure path
    fig_directory = os.path.normpath(os.path.join(check_directory, folder_name))
    # ensure path figure exists
    if not os.path.isdir(fig_directory): 
        os.makedirs(fig_directory)
    
    if cell_ids is None: 
        cell_ids = list(rep_seq_data.keys())

    # loop over cells
    for cell_nb in tqdm(cell_ids, desc="Plotting and saving rasters for each cell"):
        # setup subplots
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            figsize=(10, 10),
        )

        # add title
        plt.suptitle(f"Cell {cell_nb}", fontsize=fontsize)

        # plot raster
        utils.plot_raster_and_psth(
            rep_seq_data[cell_nb]["spike_trains"],
            rep_seq_data[cell_nb]["psth"],
            ax_rast=axs[0],
            ax_psth=axs[1],
            seq_length=rep_seq_data[cell_nb]["repeated_sequences_times"][0][1]
                       - rep_seq_data[cell_nb]["repeated_sequences_times"][0][0],
            title=title,
            fontsize=fontsize,
        )

        # format figure
        plt.subplots_adjust(wspace=0, hspace=0)

        # save figure
        fig_file = os.path.join(fig_directory, f"Cell_{cell_nb}.{save_format}")
        if save_figures: 
            plt.savefig(fig_file, dpi=fig.dpi)
        if show_figures: 
            plt.show(block=False)
            print(f"fig_file : {fig_file}")

        # clear and close figure
        plt.clf()
        plt.close()

    return


# ------------------------------------------------------------------------------------------------------------------- #
# SPIKE TRIGGERED AVERAGE
# ------------------------------------------------------------------------------------------------------------------- #

def compute_spike_triggered_average(
    checkerboard_spikes: dict,
    checkerboard: np.ndarray,
    triggers: np.ndarray,
    nb_repeats: int,
    stimulus_frequency: int,
    check_directory: str,
    nb_frames_per_sequence: int,
    temporal_dimension: int, 
    sequence_portion: tuple = (0, 0.5),
    sta_data_filename: str = "sta_data_3D.pkl",
) -> tuple:
    """
    Compute spike-triggered averages (STAs) for all cells.

    Args:
        checkerboard_spikes: Dict mapping cell IDs to spike times
        checkerboard: Stimulus array
        triggers: Array of trigger times
        nb_repeats: Number of stimulus repetitions
        stimulus_frequency: Stimulus frequency in Hz
        check_directory: Directory for saving output
        nb_frames_per_sequence: Number of frames per sequence
        temporal_dimension: Temporal dimension to use for STA in bins
        sequence_portion: Tuple specifying which portion of sequence to use
        sta_data_filename: Filename for saved STA data

    Returns:
        Tuple of (sta_data_file path, sta_data dict)
    """

    # report status
    print("Computing STAs...")

    # initialize sta output
    sta_data = {}

    # loop over spikes of checkerboard experiment
    for cell_id, spike_times in tqdm(checkerboard_spikes.items()):
        # Get spikes on random sequences
        sta_data[cell_id] = utils.extract_from_sequence(
            spike_times,
            triggers,
            nb_repeats,
            stimulus_frequency,
            sequence_portion=sequence_portion,
            nb_frames_per_sequence=nb_frames_per_sequence,
        )

        # Compute sta of the cell
        sta_3D = utils.compute_3D_sta(
            sta_data[cell_id], 
            checkerboard, 
            nb_frames_per_sequence=nb_frames_per_sequence,
            temporal_dimension=temporal_dimension,
            cell_id=cell_id
        )

        # Adding data to the notebook dictionnary
        sta_data[cell_id]["sta_3D"] = sta_3D

    # save
    sta_data_file = os.path.normpath(os.path.join(check_directory, sta_data_filename))
    utils.save_obj(sta_data, sta_data_file)
    return sta_data_file, sta_data


def plot_one_cell_3D_spike_triggered_average(
    sta_data: dict, 
    max_frames_to_show: int = 28,
    n_frames_per_line: int = 7,
    fontsize: int = 14,
    cmap: str = "RdBu_r",
):
    """
    Interactively plot 3D STA for one user-selected cell.

    Displays all temporal frames of the STA in a grid.

    Args:
        sta_data: Dictionary with STA data
        max_frames_to_show: Maximum number of temporal frames to display
        n_frames_per_line: Number of frames to show per line in the grid    
        fontsize: Font size for titles
        cmap: Colormap for STA visualization
    """
    # report number of cells
    print(f"Total : {len(sta_data)} cells found")
    print(f"Cell ids: {[int(x) for x in list(sta_data.keys())][:15]}...")

    # ask user to input a cell
    cell_id = int(input("Select a cell id: "))
    if cell_id not in sta_data:
        print(f"Error: Cell ID {cell_id} not found in STA data. Please select a valid cell ID.")
        return

    # plot
    sta = sta_data[cell_id]["sta_3D"]
    vrange = np.max(np.abs(sta))
    vmin, vmax = -1*vrange, vrange
    n_frames_to_show = min(max_frames_to_show, sta.shape[0])
    ncols = n_frames_per_line
    nrows = int(np.ceil(n_frames_to_show / ncols))
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))
    gs = GridSpec(nrows, ncols, figure=fig)
    for i in range(n_frames_to_show):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(sta[i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"f: {i}", fontsize=fontsize)
        ax.set_axis_off()
    fig.suptitle(f"Cell {cell_id} - 3D Spike-Triggered Average (shape: {sta.shape})", fontsize=fontsize)
    plt.show(block=False)
    plt.close(fig)


def analyse_all_stas(
    sta_data: dict, 
    directory: str,
    data_filename: str = "sta_data_analysed.pkl",
    method: str = "tom"
) -> dict:
    """
    Analyze STAs to extract receptive field properties, add it to the sta_data dictionary and save it.

    Args:
        sta_data: Dictionary containing STA data for each cell {cell_id: {'sta_3D': np.array (nT, nY, nX), ...}, ...}
        directory: Directory to save analyzed data
        data_filename: Filename for saved analyzed data
        method: Method to use for receptive field analysis, options are "guilhem" (default) or "gaussian_fit"

    Returns:
        sta_data: Updated dictionary with added 'sta_analysis' key for each cell containing analysis results:    
                - "Spatial": 2D numpy array representing the spatial STA.
                - "Temporal": 1D numpy array representing the temporal STA.
                - "EllipseCoor": List of parameters of the fitted ellipse (amp, x0, y0, sigma_x, sigma_y, rot_angle) in pxs.
                - "Cell_delay": Time bin corresponding to the spatial STA. 
                - "FittedEllipse": Boolean indicating whether the ellipse fitting was successful or if default parameters were returned due to an error.
        
    """

    # loop over cells, get and store rf analysis in sta_data
    for cell_id in tqdm(sta_data.keys(), desc="Fitting ellipses on STAs"):
        sta_3D = sta_data[cell_id]["sta_3D"]
        sta_data[cell_id]["sta_analysis"] = utils.rf_analysis(sta_3D, cell_id, method=method)

    # save file
    fitted_file = os.path.normpath(os.path.join(directory, data_filename))
    utils.save_obj(sta_data, fitted_file)
    return sta_data

def extend_sta_analysis_to_physical_units(
    sta_data_analysed: dict,
    pixels_per_check: int,
    pxl_size_dmd_um: float,
    sta_frequency: float,
    directory: str,
    data_filename: str = "sta_data_analysed_extended.pkl",
) -> dict:
    
    """
    Extend STA analysis results to physical units (micrometers for spatial properties, seconds for temporal properties) 
    and save the updated data.

    Args:
        sta_data_analysed: Dictionary containing STA data and analysis for each cell {cell_id: {'sta_analysis': dict, ...}, ...}
        pixels_per_check: Number of pixels per checkerboard square
        pxl_size_dmd_um: Size of one pixel in micrometers on the DMD
        sta_frequency: Frequency used STA computation (in Hz)
        directory: Directory to save updated data
        data_filename: Filename for saved updated data
    Returns:
        sta_data_analysed: Updated dictionary with added 'sta_analysis' key for each cell containing analysis results in physical units:
            - "Spatial_px_size_um": Pixel size in micrometers for the spatial STA.
            - "EllipseCoor_um": List of parameters of the fitted ellipse (amp, x0, y0, sigma_x, sigma_y, rot_angle) in micrometers.
            - "TemporalTimeVector_s": 1D numpy array representing the temporal STA time vector in seconds.
            - "TemporalFreq_s": Frequency corresponding to the temporal STA in Hz.
            - "Cell_delay_s": Time in seconds corresponding to the spatial STA.
    """
    
    sta_pixel_size_um = pxl_size_dmd_um * pixels_per_check
    sta_time_bin_s = 1 / sta_frequency
    
    for cell_id in tqdm(sta_data_analysed.keys(), desc="Extending STA analysis to physical units"):
        sta_data_analysed[cell_id]["sta_analysis"]["Spatial_px_size_um"] = sta_pixel_size_um

        ellipse_params = sta_data_analysed[cell_id]["sta_analysis"]["EllipseCoor"]
        ellipse_params_um = utils.convert_ellipse_params_to_physical_units(ellipse_params, sta_pixel_size_um)
        sta_data_analysed[cell_id]["sta_analysis"]["EllipseCoor_um"] = ellipse_params_um

        temporal_sta = sta_data_analysed[cell_id]["sta_analysis"]["Temporal"]
        temporal_sta_time_vector = utils.get_temporal_sta_time_vector(temporal_sta, sta_time_bin_s)
        sta_data_analysed[cell_id]["sta_analysis"]["TemporalTimeVector_s"] = temporal_sta_time_vector
        sta_data_analysed[cell_id]["sta_analysis"]["TemporalFreq_s"] = 1 / sta_time_bin_s

        cell_delay = sta_data_analysed[cell_id]["sta_analysis"]["Cell_delay"]
        cell_delay_s = utils.get_cell_delay_time(cell_delay, temporal_sta, sta_time_bin_s)
        sta_data_analysed[cell_id]["sta_analysis"]["Cell_delay_s"] = cell_delay_s

    # save file
    fitted_file = os.path.normpath(os.path.join(directory, data_filename))
    utils.save_obj(sta_data_analysed, fitted_file)
    return sta_data_analysed

def plot_all_stas(sta_data: dict, 
                  cell_ids: list = None,
                  fontsize: int = 35,
                  show_labels: bool = False,
                  add_fitted_indicator: bool = True,
                  border_width: int = 2,
                  level_factor: float = 0.4,
                  order_by_property: str = None):
    """
    Plot all STAs in a grid and save the figure.

    Args:
        sta_data: Dictionary containing STA data for each cell {cell_id: {'sta_3D': np.array (nT, nY, nX), ...}, ...}
        cell_ids: List of cell IDs to plot (if None, plots all cells)
        fontsize: Font size for titles and labels
        show_labels: Boolean indicating whether to show axis labels or not
        add_fitted_indicator: Boolean indicating whether to add a colored border indicating the quality of the ellipse fit
        border_width: Width of the border to indicate ellipse fit quality
        level_factor: Float factor to apply to the ellipse level when plotting the ellipse contour (default: 0.4, meaning the contour will be plotted at 40% of the ellipse amplitude)
        order_by_property: String specifying a property from sta_analysis to order the cells by before plotting ("rf_diameter", "snr", "amp")

    """
    if cell_ids is None:
        cell_ids = list(sta_data.keys())
    size = int(math.sqrt(len(cell_ids))) + 1

    # setup subplots
    good_color = "green"
    not_good_color = "orange"
    not_fitted_color = "red"
    good_count = 0    
    not_good_count = 0

    if add_fitted_indicator or order_by_property is not None:
        min_amp = 1e-3  # a.u.
        min_rf_snr = 3.5  # signal/noise (a.u.)
        for cid in tqdm(cell_ids, desc="Checking RF fit quality for each cell"):
            spatial_sta = sta_data[cid]["sta_analysis"]["Spatial"]
            ellipse_params = sta_data[cid]["sta_analysis"]["EllipseCoor"]
            invalid_coords = [(0, 0)]  # px
            min_sigma = 0.01  # px
            min_rf_area = 0.01  # px^2
            min_rf_diameter = 0.1  #px
        
            sta_data[cid]["sta_analysis"]['checkRF'] = utils.check_rf_fit(
                spatial_sta,
                ellipse_params,
                min_amp,
                invalid_coords,
                min_sigma,
                min_rf_area,
                min_rf_diameter,
                min_rf_snr,
                level_factor=level_factor,
                )

    if order_by_property is not None:
        assert order_by_property in sta_data[cell_ids[0]]["sta_analysis"]['checkRF'], f"Error: invalid order_by_property '{order_by_property}', must be one of {list(sta_data[cell_ids[0]]['sta_analysis']['checkRF'].keys())}"
        cell_ids.sort(key=lambda cid: sta_data[cid]["sta_analysis"]['checkRF'][order_by_property], reverse=True)  # sort in descending order of the chosen property

    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(50, 50))
    for i in tqdm(range(size**2), desc="Plotting STAs for all cells"):
        ax = axs[(i // size), i % size]
        if i < len(cell_ids):
            spatial_sta = sta_data[cell_ids[i]]["sta_analysis"]["Spatial"]
            ny, nx = spatial_sta.shape
            vrange = np.max(np.abs(spatial_sta))
            ax.imshow(spatial_sta, vmin=-1*vrange, vmax=vrange, cmap="RdBu_r")
            ax.set_title(f"C{cell_ids[i]}", fontsize=fontsize)
            
            if show_labels: 
                ax.set_xlabel(f"{nx} px", fontsize=fontsize)
                ax.set_ylabel(f"{ny} px", fontsize=fontsize)

            if add_fitted_indicator:
                fitted = sta_data[cell_ids[i]]["sta_analysis"]["FittedEllipse"]
                good = sta_data[cell_ids[i]]["sta_analysis"]["checkRF"]["good_rf"]
                if good:
                    border_color = good_color
                    good_count += 1
                elif not good and fitted:  
                    border_color = not_good_color
                    not_good_count += 1
                else: 
                    border_color = not_fitted_color

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)
        else:
            ax.set_visible(False)

    fitted_count = sum(1 for cell_id in cell_ids if sta_data[cell_id]["sta_analysis"]["FittedEllipse"])
    unfitted_count = sum(1 for cell_id in cell_ids if not sta_data[cell_id]["sta_analysis"]["FittedEllipse"])
    
    title = (f"Spatial STAs for {len(cell_ids)} cells\n"
             f"{fitted_count} fitted ellipse (of which: {good_count} good ({good_color}), {not_good_count} not good ({not_good_color}))\n"
             f"{unfitted_count} failed ellipse fitting ({not_fitted_color})\n")

    # format and close
    plt.tight_layout()
    # plt.suptitle(title, fontsize=fontsize, fontweight="bold")
    print(f"\n{title}")
    plt.show(block=False)
    plt.close("all")
    return None

def plot_sta_fitted_with_ellipse(
        sta_data: dict, 
        check_directory: str,
        folder_name: str = "Stas_figs",
        cell_ids: list = None,
        fontsize: int = 14,
        show_figures: bool = False,
        add_raster_plot: bool = False,
        rep_seq_data: dict = None,
        level_factor: float = 0.4,
        xdim: float = 6,
        ydim: float = 4,
        save_format: str = "png",
):
    """
    Generate single-cell figures showing the STA and ellipse fitting for all cells.

    Args:
        sta_data: Dictionary containing STA data and analysis for each cell {cell_id: {'sta_analysis': dict, ...}, ...}
        check_directory: Directory where to save the figures
        folder_name: Name of the subfolder to save the figures in
        cell_ids: List of cell IDs to plot (if None, plots all cells)
        fontsize: Font size for titles and labels
        show_figures: Boolean indicating whether to display figures interactively
        add_raster_plot: Boolean indicating whether to add raster plot to the figure (requires rep_seq_data)
        rep_seq_data: Dictionary containing extracted responses for each cell (from extract_all_cell_responses_to_repeated_sequences), required if add_raster_plot is True
        level_factor: Float factor to apply to the ellipse level when plotting the ellipse contour (default: 0.4, meaning the contour will be plotted at 40% of the ellipse amplitude)
        xdim: horizontal dimension of the figure in inches
        ydim: vertical dimension of the figure in inches
        save_format: String indicating the format to save the figures in (e.g., "png", "jpg", "svg")
    """

    # check if raster data is provided when add_raster_plot is True
    if add_raster_plot and rep_seq_data is None:
        raise ValueError("rep_seq_data must be provided when add_raster_plot is True")

    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory, folder_name))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    if cell_ids is None:
        cell_ids = list(sta_data.keys())

    # figure params
    nrows = 1
    ncols = 2 + (1 if add_raster_plot else 0)
    line_width = 2
    fontsize_labels = fontsize - 2

    # loop over all cells
    for cell_id in tqdm(cell_ids, desc="Generating STA and ellipse fitting figures for each cell"):
        # get cell sta data
        sta_analysis = sta_data[cell_id]["sta_analysis"]

        # setup figure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(xdim * ncols, ydim * nrows))
        plt.suptitle(f"Cell {cell_id}", fontsize=fontsize, fontweight="bold")

        # plot spatial sta with ellipse
        ax = axs[0]
        spatial_sta = sta_analysis["Spatial"]
        ny, nx = spatial_sta.shape
        title = f"Spatial STA ({'x' if not sta_analysis['FittedEllipse'] else '✓'} fitted)"
        if "Spatial_px_size_um" in sta_analysis:
            pixel_size_um = sta_analysis["Spatial_px_size_um"]
            xlabel = f"{nx} px ({nx*pixel_size_um:.0f} µm)"
            ylabel = f"{ny} px ({ny*pixel_size_um:.0f} µm)"
        else:
            xlabel = f"{nx} px"
            ylabel = f"{ny} px"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ellipse_params_px = sta_analysis["EllipseCoor"]
        amp, x0_px, y0_px, sigma_x_px, sigma_y_px, rot_angle = ellipse_params_px

        rf_diameter_px = np.nan
        rf_area_px2 = np.nan
        rf_poly_area_px2 = np.nan
        x0_um, y0_um, sigma_x_um, sigma_y_um = [np.nan]*4
        rf_diameter_um = np.nan
        rf_area_um2 = np.nan
        snr1, snr2, snr3 = [np.nan]*3
        add_physical_units = False
        if sta_analysis['FittedEllipse']:
            rf_diameter_px = utils.ellipse_diameter(ellipse_params_px, method="circle_approx")
            rf_area_px2 = utils.ellipse_area(ellipse_params_px, method="formula")
            rf_poly_area_px2 = utils.ellipse_area(ellipse_params_px, method="polygon", level_factor=level_factor, spatial_sta_shape=spatial_sta.shape)
            snr1 = utils.rf_snr(spatial_sta, ellipse_params_px, method='peak_std', level_factor=level_factor)
            snr2 = utils.rf_snr(spatial_sta, ellipse_params_px, method='weighted', level_factor=level_factor)
            snr3 = utils.rf_snr(spatial_sta, ellipse_params_px, method='binary_mask', level_factor=level_factor)
            if "EllipseCoor_um" in sta_analysis:
                ellipse_params_um = sta_analysis["EllipseCoor_um"]
                _, x0_um, y0_um, sigma_x_um, sigma_y_um, _ = ellipse_params_um
                rf_diameter_um = utils.ellipse_diameter(ellipse_params_um, method="circle_approx")
                rf_area_um2 = utils.ellipse_area(ellipse_params_um, method="formula")
                add_physical_units = True

        text = f"Amplitude {amp:.3f} a.u.\n"
        if add_physical_units:
            text += (f"Center xy ({x0_um:.0f}, {y0_um:.0f}) µm [({x0_px:.1f}, {y0_px:.1f}) px]\n"
                    f"Var xy ({sigma_x_um:.0f}, {sigma_y_um:.0f}) µm [({sigma_x_px:.1f}, {sigma_y_px:.1f}) px]\n"
                    f"Rotation {rot_angle:.1f}°\n\n"
                    f"RF diameter {rf_diameter_um:.0f} µm [{rf_diameter_px:.1f} px]\n"
                    f"RF area {rf_area_um2:.0f} µm² [{rf_area_px2:.1f} px²]\n"
                    )
        else:
            text += (f"Center xy ({x0_px:.1f}, {y0_px:.1f}) px\n"
                    f"Var xy ({sigma_x_px:.1f}, {sigma_y_px:.1f}) px\n"
                    f"Rotation {rot_angle:.1f}°\n\n"
                    f"RF diameter {rf_diameter_px:.1f} px\n"
                    f"RF area {rf_area_px2:.1f} px²\n"
                    )
        text += f"RF poly area {rf_poly_area_px2:.1f} px²\n"
        text += "\n"
        text += f"SNR (peak/std): {snr1:.2f}\n"
        text += f"SNR (gauss prj/resid): {snr2:.2f}\n"
        text += f"SNR (in/out): {snr3:.2f}\n"
        utils.plot_sta(ax, spatial_sta, sta_analysis["EllipseCoor"], 
                       level_factor=level_factor, color="yellow", alpha=1, lw=line_width, linestyles="solid")
        ax.set_title(title, fontsize=fontsize)

        ax.text(-0.25, 1, text, transform=ax.transAxes, fontsize=fontsize_labels, va='top', ha='right')

        # plot temporal sta
        ax = axs[1]
        ax.set_title("Temporal STA", fontsize=fontsize)
        tsta_y = sta_analysis["Temporal"]
        peak_value = tsta_y[sta_analysis["Cell_delay"]] if sta_analysis["Cell_delay"] is not None and not np.isnan(sta_analysis["Cell_delay"]) else None
        if "TemporalTimeVector_s" in sta_analysis and "TemporalFreq_s" in sta_analysis and "Cell_delay_s" in sta_analysis:
            tsta_x = sta_analysis["TemporalTimeVector_s"]
            cell_delay = sta_analysis["Cell_delay_s"]
            xlabel = "Time (s)"
            text = f"Frequency: {sta_analysis['TemporalFreq_s']:.1f} Hz\nCell delay: {cell_delay:.3f} s"
        else:
            tsta_x = np.arange(len(tsta_y))
            cell_delay = sta_analysis["Cell_delay"]
            xlabel = "Time bins"
            text = f"Frequency: N/A\nCell delay: {cell_delay} bins"
        if peak_value is None or peak_value >= 0:
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize_labels, va='top', ha='left')
        else:
            ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=fontsize_labels, va='bottom', ha='left')

        ax.plot(tsta_x, tsta_y, lw=line_width)
        if cell_delay is not None and not np.isnan(cell_delay):
            ax.axvline(cell_delay, color="gray", lw=0.5, ls="--")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel("STA amplitude", fontsize=fontsize)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        if add_raster_plot:
            ax = axs[2]
            ax.eventplot(rep_seq_data[cell_id]["spike_trains"])
            ax.set_title("Raster plot", fontsize=fontsize)
            ax.set_xlabel("Time (s)", fontsize=fontsize)
            ax.set_ylabel(f"{len(rep_seq_data[cell_id]['spike_trains'])} repetitions", fontsize=fontsize)
            ax.set_aspect('auto')
            ax.set_ylim(0, len(rep_seq_data[cell_id]["spike_trains"]))

        for ax in axs: 
            ax.tick_params(axis="both", which="major", labelsize=fontsize_labels)

        fig.tight_layout()
        fig_file = os.path.join(fig_directory, f"Cell_{cell_id}.{save_format}")

        if show_figures:
            plt.show()

        # save figure
        plt.savefig(fig_file, dpi=fig.dpi)
        # close figure to free memory
        plt.close(fig)
    return None



# ------------------------------------------------------------------------------------------------------------------- #
# OLD FUNCTIONS
# ------------------------------------------------------------------------------------------------------------------- #

def plot_sta_fitted_with_ellipse_by_tom(
    raster_data: dict, 
    cells_id: list, 
    cells_to_plot: list, 
    check_directory: str,
    save_format: str = "png"
):
    """
    Plot comprehensive analysis including raster, fitted STA, and temporal profile.

    Creates three-panel plots for each cell showing:
    1. Raster plot
    2. Fitted ellipse on spatial STA
    3. Temporal STA profile

    Args:
        raster_data: Dictionary with raster data
        cells_id: List of all cell IDs to process
        cells_to_plot: List of cell IDs to display interactively
        check_directory: Directory containing fitted STA data
        save_format: String indicating the format to save the figures in (e.g., "png", "jpg", "svg")
    """

    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory, r"Stas_figs"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    sta_data = np.load(
        os.path.join(check_directory, "sta_data_3D_fitted.pkl"), allow_pickle=True
    )

    # loop over cells
    for cell_id in tqdm(cells_id[0:]):
        # setup subplots
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

        plt.suptitle(f"Cell {cell_id}")
        sta = sta_data[cell_id]["center_analyse"]
        ax = axs[0]
        ax.eventplot(raster_data[cell_id]["spike_trains"])
        ax.set(title="Raster plot", ylabel="N Repetitions")

        ax = axs[1]
        ax.set(title="Fitted Ellipse")

        try:
            ax = utils.plot_sta_tom(ax, sta["Spatial"], sta["EllipseCoor"])
        except:
            ax.imshow(sta_data[cell_id]["center_analyse"]["Spatial"])

        ax = axs[2]
        ax.set(title="Temporal STA")
        ax.plot(sta["Temporal"])
        ax.set_ylim([-1, 1])

        fig_file = os.path.join(fig_directory, f"Cell_{cell_id}.{save_format}")
        plt.savefig(fig_file, dpi=fig.dpi)

        # plot selected cells only
        if cell_id in cells_to_plot:
            plt.show()

        # close figure to free memory
        plt.clf()
        plt.close(fig)

def check_all_stas(sta_data_analysed, nb_pixels_per_check):
    """
    The Goal of this cell is to quantify the number of STAs in your data
    Tuned to a 40x40 with square of 15 pixels checkerboard.
    Maybe, you need to tune it for your specific checkerboard
    See with Tom Quetu if needed for the tuning
    """
    sta_quantification_dict={}
    for cell_id in tqdm(sta_data_analysed.keys(), desc="Checking all STAs"):
        fig, ax = plt.subplots(nrows = 1,ncols = 1, figsize=(7,7))

        plt.suptitle(f'Cell {cell_id}')
        spatial_sta = sta_data_analysed[cell_id]["sta_analysis"]['Spatial']
        ellipse_params = sta_data_analysed[cell_id]["sta_analysis"]['EllipseCoor']
        sta_quantification = utils.check_presence_STA(
            spatial_sta,
            ellipse_params,
            nb_pixels_per_check,
            tresh_snr=2.75, 
            level_factor=0.2
            )
        try:
            ax = utils.plot_sta_tom(ax, spatial_sta, ellipse_params)
        except:
            ax.imshow(spatial_sta)

        color_cadr='r'
        if sta_quantification[0]==1:
            color_cadr='g'
            ax.set(title='STA with a SNR of ' + str(sta_quantification[1]))
        elif sta_quantification[0]==0.1:
            ax.set(title='ellipse not centered')
        elif sta_quantification[0]==0.2:
            ax.set(title='no ellipse fitted')
        elif sta_quantification[0]==0.3:
            ax.set(title='diameter too big or too small, diameter: ' + str(sta_quantification[1]) +'µm')
        elif sta_quantification[0]==0.4:
            ax.set(title='No STA because the SNR is too small, SNR:' + str(sta_quantification[1]))

        ax.spines['top'].set(lw=6, color=color_cadr)
        ax.spines['bottom'].set(lw=6, color=color_cadr)
        ax.spines['left'].set(lw=6, color=color_cadr)
        ax.spines['right'].set(lw=6, color=color_cadr)

        sta_quantification_dict[cell_id]=sta_quantification[0]
        plt.show()
        plt.close()
    return sta_quantification_dict
            
# ------------------------------------------------------------------------------------------------------------------- #