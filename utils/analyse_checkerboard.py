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
    stimulus_frequency = int(
        input("Select stimulus frequency (Hz, usually in stimulus filename): ")
    )
    nb_checks_x = int(
        input("Select number of checks on x (sq, usually in stimulus filename): ")
    )
    nb_checks_y = int(
        input("Select number of checks on y (sq, usually in stimulus filename): ")
    )
    nb_pixels_per_check = int(
        input("Select number of pixels per check (px, usually in stimulus filename): ")
    )

    return stimulus_frequency, nb_checks_x, nb_checks_y, nb_pixels_per_check


def get_all_inputs_for_checkerboard_analysis(
    params: ModuleType,
    is_swn: bool = False,
) -> tuple[int, str, float, int, int, int, str]:
    """
    Get all input parameters for a checkerboard (or SWN) experiment analysis.

    Interactively prompts for the recording and stimulus frequency, and for the
    checkerboard the number/size of checks too. Creates the analysis directory.

    Args:
        params: params module with recording_names and output_directory.
        is_swn: if True, this is a Shifting-White-Noise recording — the check
            count/size questions are skipped (they don't apply to SWN; its spatial
            resolution comes from the .bin frames and the down-sampling shift).

    Returns:
        Tuple containing:
            - recording_number: Selected recording index
            - recording_name: Selected recording name
            - stimulus_frequency: Stimulus frequency in Hz
            - nb_checks_x: Number of checks in x (None for SWN)
            - nb_checks_y: Number of checks in y (None for SWN)
            - nb_pixels_per_check: Number of pixels per check (None for SWN)
            - check_directory: Path to analysis output directory
    """
    stim_label = "SWN" if is_swn else "checkerboard"
    recording_number, recording_name = utils.prompt_user_for_recording(
        params.recording_names, stim_label
    )
    if is_swn:
        # SWN: only the stimulus frequency is needed (no checks).
        stimulus_frequency = int(
            input("Select stimulus frequency (Hz, usually in stimulus filename): ")
        )
        nb_checks_x = nb_checks_y = nb_pixels_per_check = None
    else:
        stimulus_frequency, nb_checks_x, nb_checks_y, nb_pixels_per_check = (
            prompt_user_for_checkerboard_params()
        )
    check_directory = utils.create_analysis_directory(
        params.output_directory, recording_number, "Checkerboard"
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
    triggers: np.ndarray, params: ModuleType, stimulus_frequency: float
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
            mea=params.MEA,
        )

    return checkerboard


def load_checkerboard_data(
    params: ModuleType,
    check_directory: str,
    checkerboard_name: str,
    nb_checks_x: int,
    nb_checks_y: int,
    stimulus_frequency: float,
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
        triggers_path, params.fs, verbose=True
    )
    cells_id, checkerboard_spikes = utils.load_spike_times(
        checkerboard_name, params.output_directory, params.exp
    )
    nb_repeats, _ = calculate_checkerboard_experiment_stats(
        stim_onsets, params, stimulus_frequency
    )
    checkerboard = load_or_create_checkerboard_stimulus(
        nb_repeats, nb_checks_x, nb_checks_y, check_directory, params
    )
    print(f"Checkerboard stimulus shape: {checkerboard.shape}")

    print(
        f"\nTotal : {len(checkerboard_spikes.keys())} neurons loaded\nCell ids: {[int(x) for x in cells_id]}\n"
    )

    return checkerboard_spikes, stim_onsets, nb_repeats, cells_id, checkerboard


# ------------------------------------------------------------------------------------------------------------------- #
# SWN (Shifting White Noise) — alternative stimulus. Reuses the whole checkerboard STA
# pipeline; only the stimulus reconstruction and one decorrelation step differ.
# ------------------------------------------------------------------------------------------------------------------- #


def load_swn_stimulus(
    bin_path: str,
    vec_path: str,
    rig_id: int,
    shift_x: int,
    shift_y: int,
    sigma: float = 5.0,
) -> tuple:
    """
    Reconstruct a Shifting-White-Noise (SWN) stimulus from its .bin (raw frames) and
    .vec files, and compute the stimulus covariance used later to whiten the STA.

    Unlike the checkerboard (drawn from a white binary source), SWN frames are
    spatially correlated, so the spatial STA must be decorrelated by the inverse of
    this covariance afterwards (see decorrelate_spatial_stas).

    Args:
        bin_path: path to the SWN .bin file (raw noise frames, read via utils.binfile.BinFile).
        vec_path: path to the SWN .vec file (its header column 1 gives the total frame count).
        rig_id: MEA / rig id (params.MEA). Its DMD geometry, polarity and optical
            transform are read from params.rig_params; rigs that are not implemented
            /tested raise a clear error.
        shift_x: spatial down-sampling step in x (pixels).
        shift_y: spatial down-sampling step in y (pixels).
        sigma: value added to the covariance diagonal for numerical stability.

    Returns:
        stimulus: np.ndarray (n_frames, H, W) of the down-sampled, unrepeated SWN frames.
        C_I: np.ndarray (H*W, H*W) regularised stimulus covariance matrix.
    """
    from .binfile import BinFile

    vec_data = np.loadtxt(vec_path)
    vec_trigs, vec_header = vec_data[1:], vec_data[0]
    # Only the first half of the SWN frames are unrepeated (used to build the STA).
    num_unrepeated_frames = int(vec_header[1] / 2)

    bin_obj = BinFile(
        bin_path, 0, 0, rig_id, mode="r"
    )  # frame size is read from the .bin header
    frames = []
    for vec in tqdm(vec_trigs, desc="Reconstructing SWN stimulus"):
        frame_index = int(vec[1])
        if frame_index < num_unrepeated_frames:
            frame = bin_obj.read_frame(frame_index)
            frames.append(frame[::shift_x, ::shift_y] / frame.max())
    bin_obj.close()
    stimulus = np.array(frames)
    print(
        f"SWN stimulus reconstructed: {stimulus.shape[0]} frames of {stimulus.shape[1]}x{stimulus.shape[2]}"
    )

    # Stimulus covariance (for decorrelating the STA); regularised on the diagonal.
    print("Computing SWN stimulus covariance matrix...")
    stimulus_matrix = stimulus.reshape(len(stimulus), -1)
    cov = np.cov(stimulus_matrix.T)
    C_I = cov + np.eye(cov.shape[0]) * sigma
    return stimulus, C_I


def load_swn_data(
    params: ModuleType,
    swn_recording_name: str,
    stimulus_frequency: float,
) -> tuple:
    """
    Load everything needed for an SWN analysis: triggers, spikes, the reconstructed
    SWN stimulus and its covariance matrix.

    Triggers and spikes are loaded exactly like the checkerboard path; only the
    stimulus comes from the SWN .bin/.vec (via load_swn_stimulus) and an extra
    covariance matrix C_I is returned for the later STA decorrelation.

    Args:
        params: params module (uses triggers_directory, exp, fs, output_directory, MEA,
            stim_directory, swn_bin_file, swn_vec_file, swn_shift_x, swn_shift_y,
            swn_cov_regularization).
        swn_recording_name: the SWN recording name (to locate its triggers/spikes).
        stimulus_frequency: stimulus frequency in Hz.

    Returns:
        swn_spikes, stim_onsets, nb_repeats, cells_id, stimulus, C_I
    """
    triggers_path = os.path.normpath(
        os.path.join(
            params.triggers_directory, f"{params.exp}_{swn_recording_name}_triggers.pkl"
        )
    )
    print(f"Loading triggers from:\t {triggers_path}")
    stim_onsets = utils.load_stim_onset_from_triggers_path(
        triggers_path, params.fs, verbose=True
    )
    cells_id, swn_spikes = utils.load_spike_times(
        swn_recording_name, params.output_directory, params.exp
    )
    nb_repeats, _ = calculate_checkerboard_experiment_stats(
        stim_onsets, params, stimulus_frequency
    )

    # Resolve the SWN .bin/.vec from the stimulus folder (prompts if the exact file is
    # not there), the same way the other analyses find their .vec files.
    swn_bin_path = utils.find_vec_file(params.swn_bin_file, params.stim_directory)
    swn_vec_path = utils.find_vec_file(params.swn_vec_file, params.stim_directory)
    stimulus, C_I = load_swn_stimulus(
        swn_bin_path,
        swn_vec_path,
        params.MEA,
        params.swn_shift_x,
        params.swn_shift_y,
        params.swn_cov_regularization,
    )

    print(
        f"\nTotal : {len(swn_spikes)} neurons loaded\nCell ids: {[int(x) for x in cells_id]}\n"
    )
    return swn_spikes, stim_onsets, nb_repeats, cells_id, stimulus, C_I


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
    sequence_portion: tuple = (0.5, 1),
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
    for cell_id, spike_times in tqdm(
        checkerboard_spikes.items(),
        desc="Extracting responses to repeated sequence for each cell",
    ):
        output_data[cell_id] = utils.extract_from_sequence(
            spike_times,
            triggers,
            nb_repeats,
            stim_frequency=stimulus_frequency,
            sequence_portion=sequence_portion,
            nb_frames_per_sequence=nb_frames_per_sequence,
        )

    # check data
    assert set(output_data.keys()) == set(cells_id), (
        "Error in extracting data: Cell IDs in output data do not match expected cell IDs"
    )
    for cell_id in cells_id:
        assert (
            "spike_times" in output_data[cell_id]
            and output_data[cell_id]["spike_times"].ndim == 1
        ), (
            f"Error in extracting data: 'spike_times' key missing or None for cell ID {cell_id}"
        )
        assert (
            "repeated_sequences_times" in output_data[cell_id]
            and isinstance(output_data[cell_id]["repeated_sequences_times"], list)
            and len(output_data[cell_id]["repeated_sequences_times"]) == nb_repeats
        ), (
            f"Error in extracting data: 'repeated_sequences_times' key missing or not a list for cell ID {cell_id}"
        )
        assert (
            "spike_trains" in output_data[cell_id]
            and isinstance(output_data[cell_id]["spike_trains"], list)
            and len(output_data[cell_id]["spike_trains"]) == nb_repeats
        ), (
            f"Error in extracting data: 'spike_trains' key missing or not a list for cell ID {cell_id}"
        )
        assert (
            "counted_spikes" in output_data[cell_id]
            and output_data[cell_id]["counted_spikes"].ndim == 2
            and output_data[cell_id]["counted_spikes"].shape[0] == nb_repeats
        ), (
            f"Error in extracting data: 'counted_spikes' key missing or not a 2D array for cell ID {cell_id}"
        )
        nbins = output_data[cell_id]["counted_spikes"].shape[1]
        assert (
            "psth" in output_data[cell_id]
            and output_data[cell_id]["psth"].ndim == 1
            and output_data[cell_id]["psth"].shape[0] == nbins
        ), (
            f"Error in extracting data: 'psth' key missing or not a 1D array for cell ID {cell_id}"
        )
    return output_data


def plot_all_rasters(
    rep_seq_data: dict,
    cells_id: list,
    fontsize: int = 35,
    show_labels: bool = False,
    plotting: bool = True,
):
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


def plot_raster_and_psth(
    spike_trains: list,
    psth: list,
    ax_rast: plt.Axes,
    ax_psth: plt.Axes,
    seq_length: float,
    t0: float = 0,
    title: str = "Raster plot",
    fontsize: int = 12,
    fontsize_labels: int = None,
):
    """
    Plot a raster plot and PSTH for given data in given axes.

    Args:
        spike_trains: List of spike trains (one per repetition). Length should be equal to the number of repetitions of the sequence.
        psth: Peri-stimulus time histogram values (firing rate in spikes/s) for each time bin. Length should be equal to the number of time bins used to extract the sequence response.
        ax_rast: Matplotlib axis for the raster plot.
        ax_psth: Matplotlib axis for the PSTH.
        seq_length: Duration of the sequence in seconds.
        t0: Start time of the sequence (default 0).
        title: Title for the raster plot (default "Raster plot").
        fontsize: Font size for titles and labels (default 12).
        fontsize_labels: Font size for axis labels (default fontsize-2).

    Returns:
        bin_values: Time values corresponding to the center of each PSTH bin in seconds.
        bin_width: Width of each PSTH bin in seconds.
    """
    if fontsize_labels is None:
        fontsize_labels = fontsize - 2
    ax_rast.eventplot(spike_trains)
    # Even/odd reliability of the repeated responses, shown in the raster title.
    reliability = utils.even_odd_reliability(
        spike_trains, len(psth), (t0, t0 + seq_length)
    )
    rel_txt = "n/a" if np.isnan(reliability) else f"{reliability:.2f}"
    ax_rast.set_title(f"{title}  (reliability r = {rel_txt})", fontsize=fontsize)
    ax_rast.set_ylabel("n repetition", fontsize=fontsize)

    nbins = len(psth)
    bin_edges = np.linspace(t0, t0 + seq_length, nbins)
    bin_width = np.diff(bin_edges)[0]
    bin_values = bin_edges + bin_width / 2

    ax_psth.bar(
        bin_values,
        psth,
        width=bin_width,
    )
    ax_psth.set_xlabel("Time (s)", fontsize=fontsize)
    ax_psth.set_ylabel("Firing rate (spikes/s)", fontsize=fontsize)
    for ax in [ax_rast, ax_psth]:
        ax.tick_params(axis="both", which="major", labelsize=fontsize_labels)
    return bin_values, bin_width


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
        plot_raster_and_psth(
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
            cell_id=cell_id,
        )

        # Adding data to the notebook dictionary
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
        print(
            f"Error: Cell ID {cell_id} not found in STA data. Please select a valid cell ID."
        )
        return

    # plot
    sta = sta_data[cell_id]["sta_3D"]
    vrange = np.max(np.abs(sta))
    vmin, vmax = -1 * vrange, vrange
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
    fig.suptitle(
        f"Cell {cell_id} - 3D Spike-Triggered Average (shape: {sta.shape})",
        fontsize=fontsize,
    )
    plt.show(block=False)
    plt.close(fig)


def analyse_all_stas(
    sta_data: dict,
    directory: str,
    data_filename: str = "sta_data_analysed.pkl",
    method: str = "tom",
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
                - "EllipseCoor": List of parameters of the fitted ellipse (amp, x0, y0, sigma_x, sigma_y, rot_angle) in units.
                - "Cell_delay": Time bin corresponding to the spatial STA.
                - "FittedEllipse": Boolean indicating whether the ellipse fitting was successful or if default parameters were returned due to an error.

    """

    # loop over cells, get and store rf analysis in sta_data
    for cell_id in tqdm(sta_data.keys(), desc="Fitting ellipses on STAs"):
        sta_3D = sta_data[cell_id]["sta_3D"]
        sta_data[cell_id]["sta_analysis"] = utils.rf_analysis(
            sta_3D, cell_id, method=method
        )

    # save file
    fitted_file = os.path.normpath(os.path.join(directory, data_filename))
    utils.save_obj(sta_data, fitted_file)
    return sta_data


def decorrelate_spatial_stas(sta_data: dict, C_I: np.ndarray) -> dict:
    """
    Whiten each cell's spatial STA by the inverse SWN stimulus covariance and re-fit
    the RF ellipse. This is the ONLY analysis step specific to SWN — the checkerboard
    stimulus is already white and skips it.

    Run this AFTER analyse_all_stas and BEFORE extend_sta_analysis_to_physical_units.
    The whitened RF overwrites ``sta_analysis["Spatial"]`` (the raw STA is kept under
    ``"Spatial_raw"``), so the spatial STA plot shows the decorrelated RF.

    ``EllipseCoor`` is deliberately LEFT UNCHANGED: it stays the ellipse fitted on the
    mask, exactly like the checkerboard pipeline, so SWN and checkerboard save the same
    ellipse (and everything downstream — physical units, plots, cell typing — uses it).
    The ellipse re-fitted on the whitened STA is kept separately as ``EllipseCoor_whitened``
    for reference; it is NOT used as the RF, because whitening amplifies noise and can move
    the fit off weak cells' true RF.

    Args:
        sta_data: dict from analyse_all_stas; each cell has a ``"sta_analysis"`` entry.
        C_I: regularised stimulus covariance returned by load_swn_stimulus.

    Returns:
        sta_data, updated in place.
    """
    default_ellipse = [0, 0, 0, 0.001, 0.001, 0]
    for cell_id in tqdm(sta_data.keys(), desc="Decorrelating SWN STAs"):
        analysis = sta_data[cell_id]["sta_analysis"]
        spatial = analysis["Spatial"]
        if np.max(np.abs(spatial)) == 0:  # silent cell, nothing to whiten
            continue

        # Whiten: solve C_I x = sta  (multiply the flattened RF by the inverse covariance).
        whitened = np.linalg.solve(C_I, spatial.flatten()).reshape(spatial.shape)
        analysis["Spatial_raw"] = spatial
        analysis["Spatial"] = whitened

        # Re-fit on the whitened RF and keep it as EllipseCoor_whitened, but do NOT overwrite
        # EllipseCoor (the mask fit) — SWN keeps the same ellipse as the checkerboard.
        try:
            fitting_data = utils.preprocess_fitting_standard(whitened)
            ellipse_params, _ = utils.double_gaussian_fit(fitting_data)
            analysis["EllipseCoor_whitened"] = ellipse_params
        except Exception:
            print(f"Could not re-fit the whitened ellipse for cell {cell_id}")
            analysis["EllipseCoor_whitened"] = default_ellipse
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
            - "Spatial_unit_size_um": Pixel size in micrometers for the spatial STA.
            - "EllipseCoor_um": List of parameters of the fitted ellipse (amp, x0, y0, sigma_x, sigma_y, rot_angle) in micrometers.
            - "TemporalTimeVector_s": 1D numpy array representing the temporal STA time vector in seconds.
            - "TemporalFreq_s": Frequency corresponding to the temporal STA in Hz.
            - "Cell_delay_s": Time in seconds corresponding to the spatial STA.
    """

    sta_pixel_size_um = pxl_size_dmd_um * pixels_per_check
    sta_time_bin_s = 1 / sta_frequency

    for cell_id in tqdm(
        sta_data_analysed.keys(), desc="Extending STA analysis to physical units"
    ):
        sta_data_analysed[cell_id]["sta_analysis"]["Spatial_unit_size_um"] = (
            sta_pixel_size_um
        )

        ellipse_params = sta_data_analysed[cell_id]["sta_analysis"]["EllipseCoor"]
        ellipse_params_um = utils.convert_ellipse_params_to_physical_units(
            ellipse_params, sta_pixel_size_um
        )
        sta_data_analysed[cell_id]["sta_analysis"]["EllipseCoor_um"] = ellipse_params_um

        temporal_sta = sta_data_analysed[cell_id]["sta_analysis"]["Temporal"]
        temporal_sta_time_vector = utils.get_temporal_sta_time_vector(
            temporal_sta, sta_time_bin_s
        )
        sta_data_analysed[cell_id]["sta_analysis"]["TemporalTimeVector_s"] = (
            temporal_sta_time_vector
        )
        sta_data_analysed[cell_id]["sta_analysis"]["TemporalFreq_s"] = (
            1 / sta_time_bin_s
        )

        cell_delay = sta_data_analysed[cell_id]["sta_analysis"]["Cell_delay"]
        cell_delay_s = utils.get_cell_delay_time(
            cell_delay, temporal_sta, sta_time_bin_s
        )
        sta_data_analysed[cell_id]["sta_analysis"]["Cell_delay_s"] = cell_delay_s

    # save file
    fitted_file = os.path.normpath(os.path.join(directory, data_filename))
    utils.save_obj(sta_data_analysed, fitted_file)
    return sta_data_analysed


def plot_all_stas(
    sta_data: dict,
    cell_ids: list = None,
    fontsize: int = 35,
    show_labels: bool = False,
    add_fitted_indicator: bool = True,
    border_width: int = 2,
    n_sigma: float = 2.0,
    order_by_property: str = None,
    set_axis_off: bool = True,
):
    """
    Plot all STAs in a grid and save the figure.

    Args:
        sta_data: Dictionary containing STA data for each cell {cell_id: {'sta_3D': np.array (nT, nY, nX), ...}, ...}
        cell_ids: List of cell IDs to plot (if None, plots all cells)
        fontsize: Font size for titles and labels
        show_labels: Boolean indicating whether to show axis labels or not
        add_fitted_indicator: Boolean indicating whether to add a colored border indicating the quality of the ellipse fit
        border_width: Width of the border to indicate ellipse fit quality
        n_sigma: Number of standard deviations of the fitted Gaussian at which the RF ellipse contour / SNR mask is defined (default: 2.0). Replaces the old level_factor (peak-fraction) parameter.
        order_by_property: String specifying a property from sta_analysis to order the cells by before plotting ("rf_diameter", "snr", "amp")

    """
    # A 2D Gaussian reaches exp(-n^2/2) of its peak at n standard deviations, so this is
    # the peak fraction that the internal contour/SNR helpers expect.
    level_factor = np.exp(-(n_sigma**2) / 2)

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
            invalid_coords = [(0, 0)]  # unit
            min_sigma = 0.01  # unit
            min_rf_area = 0.01  # unit^2
            min_rf_diameter = 0.1  # unit

            sta_data[cid]["sta_analysis"]["checkRF"] = utils.check_rf_fit(
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
        assert order_by_property in sta_data[cell_ids[0]]["sta_analysis"]["checkRF"], (
            f"Error: invalid order_by_property '{order_by_property}', must be one of {list(sta_data[cell_ids[0]]['sta_analysis']['checkRF'].keys())}"
        )
        cell_ids.sort(
            key=lambda cid: sta_data[cid]["sta_analysis"]["checkRF"][order_by_property],
            reverse=True,
        )  # sort in descending order of the chosen property

    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(50, 50))
    for i in tqdm(range(size**2), desc="Plotting STAs for all cells"):
        ax = axs[(i // size), i % size]
        if i < len(cell_ids):
            spatial_sta = sta_data[cell_ids[i]]["sta_analysis"]["Spatial"]
            ny, nx = spatial_sta.shape
            vrange = np.max(np.abs(spatial_sta))
            ax.imshow(spatial_sta, vmin=-1 * vrange, vmax=vrange, cmap="RdBu_r")
            ax.set_title(f"C{cell_ids[i]}", fontsize=fontsize)
            if "Spatial_unit_size_um" in sta_data[cell_ids[i]]["sta_analysis"]:
                utils.add_scalebar(
                    ax,
                    scalebar_size_um=100,
                    pixel_size_um=sta_data[cell_ids[i]]["sta_analysis"][
                        "Spatial_unit_size_um"
                    ],
                    scalebar_left_location=(0.9, 0.9),
                    nx=nx,
                    ny=ny,
                    scale_bar_color="black",
                    scale_bar_width=4,
                )

            if set_axis_off:
                ax.set_xticks([])
                ax.set_yticks([])

            if show_labels:
                ax.set_xlabel(f"{nx} unit", fontsize=fontsize)
                ax.set_ylabel(f"{ny} unit", fontsize=fontsize)

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

    fitted_count = sum(
        1 for cell_id in cell_ids if sta_data[cell_id]["sta_analysis"]["FittedEllipse"]
    )
    unfitted_count = sum(
        1
        for cell_id in cell_ids
        if not sta_data[cell_id]["sta_analysis"]["FittedEllipse"]
    )

    title = (
        f"Spatial STAs for {len(cell_ids)} cells\n"
        f"{fitted_count} fitted ellipse (of which: {good_count} good ({good_color}), {not_good_count} not good ({not_good_color}))\n"
        f"{unfitted_count} failed ellipse fitting ({not_fitted_color})\n"
    )

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
    add_spatial_mask: bool = False,
    rep_seq_data: dict = None,
    n_sigma: float = 2.0,
    xdim: float = 8,
    ydim: float = 5,
    save_format: str = "png",
    scale_bar_color: str = "black",  # STA scale bar (light background); the mask uses chartreuse
    scale_bar_width: int = 4,
    scalebar_left_location: tuple = (0.95, 0.95),
    scalebar_size_um: int = 100,
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
        n_sigma: Number of standard deviations of the fitted Gaussian at which the RF ellipse contour / SNR mask is defined (default: 2.0). Replaces the old level_factor (peak-fraction) parameter.
        xdim: horizontal dimension of the figure in inches
        ydim: vertical dimension of the figure in inches
        save_format: String indicating the format to save the figures in (e.g., "png", "jpg", "svg")
        scale_bar_color: Color of the scale bar to add to the spatial STA plot (e.g., "black", "white", "red")
        scale_bar_width: Width of the line of the scale bar in matplotlib units (e.g., 4)
        scalebar_left_location: Tuple of (x, y) coordinates in relative axes units (between 0 and 1) indicating the left end location of the scale bar on the spatial STA plot (e.g., (.9, .1) for bottom right corner)
        scalebar_size_um: Size of the scale bar in micrometers (e.g., 100 for a 100µm scale bar)
    """

    # check if raster data is provided when add_raster_plot is True
    if add_raster_plot and rep_seq_data is None:
        raise ValueError("rep_seq_data must be provided when add_raster_plot is True")

    # A 2D Gaussian reaches exp(-n^2/2) of its peak at n standard deviations, so this is
    # the peak fraction that the internal contour/SNR helpers expect.
    level_factor = np.exp(-(n_sigma**2) / 2)

    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory, folder_name))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    if cell_ids is None:
        cell_ids = list(sta_data.keys())

    # figure params
    nrows = 1
    ncols = 2 + (1 if add_raster_plot else 0) + (1 if add_spatial_mask else 0)
    line_width = 2
    fontsize_labels = fontsize - 2
    components_coords_color = "cyan"

    # loop over all cells
    for cell_id in tqdm(
        cell_ids, desc="Generating STA and ellipse fitting figures for each cell"
    ):
        # get cell sta data
        sta_analysis = sta_data[cell_id]["sta_analysis"]

        # setup figure
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(xdim * ncols, ydim * nrows)
        )
        plt.suptitle(f"Cell {cell_id}", fontsize=fontsize, fontweight="bold")
        next_ax = 0

        # plot spatial sta with ellipse
        ax = axs[next_ax]
        spatial_sta = sta_analysis["Spatial"]
        ny, nx = spatial_sta.shape
        title = (
            f"Spatial STA ({'x' if not sta_analysis['FittedEllipse'] else '✓'} fitted)"
        )
        ellipse_params_unit = sta_analysis["EllipseCoor"]
        amp, x0_unit, y0_unit, sigma_x_unit, sigma_y_unit, rot_angle = (
            ellipse_params_unit
        )
        rf_diameter_unit = np.nan
        rf_area_unit2 = np.nan
        # rf_poly_area_unit2 = np.nan
        x0_um, y0_um, sigma_x_um, sigma_y_um = [np.nan] * 4
        rf_diameter_um = np.nan
        rf_area_um2 = np.nan
        snr1, snr2, snr3 = [np.nan] * 3
        add_physical_units = False
        if sta_analysis["FittedEllipse"]:
            rf_diameter_unit = utils.ellipse_diameter(
                ellipse_params_unit, method="circle_approx"
            )
            rf_area_unit2 = utils.ellipse_area(ellipse_params_unit, method="formula")
            # rf_poly_area_unit2 = utils.ellipse_area(ellipse_params_unit, method="polygon", level_factor=level_factor, spatial_sta_shape=spatial_sta.shape)
            snr1 = utils.rf_snr(
                spatial_sta,
                ellipse_params_unit,
                method="peak_std",
                level_factor=level_factor,
            )
            snr2 = utils.rf_snr(
                spatial_sta,
                ellipse_params_unit,
                method="weighted",
                level_factor=level_factor,
            )
            snr3 = utils.rf_snr(
                spatial_sta,
                ellipse_params_unit,
                method="binary_mask",
                level_factor=level_factor,
            )
            if "EllipseCoor_um" in sta_analysis:
                ellipse_params_um = sta_analysis["EllipseCoor_um"]
                _, x0_um, y0_um, sigma_x_um, sigma_y_um, _ = ellipse_params_um
                rf_diameter_um = utils.ellipse_diameter(
                    ellipse_params_um, method="circle_approx"
                )
                rf_area_um2 = utils.ellipse_area(ellipse_params_um, method="formula")
                add_physical_units = True
        text = ""
        text += f"Ellipse plotted at {n_sigma:g}σ\n\n"
        text += f"Amplitude {amp:.3f} a.u.\n"
        if add_physical_units:
            text += (
                f"Center xy ({x0_um:.0f}, {y0_um:.0f}) µm [({x0_unit:.1f}, {y0_unit:.1f}) unit]\n"
                f"Sigma xy ({sigma_x_um:.0f}, {sigma_y_um:.0f}) µm [({sigma_x_unit:.1f}, {sigma_y_unit:.1f}) unit]\n"
                f"Rotation {rot_angle:.1f}°\n\n"
                f"RF diameter {rf_diameter_um:.0f} µm [{rf_diameter_unit:.1f} unit]\n"
                f"RF area {rf_area_um2:.0f} µm² [{rf_area_unit2:.1f} unit²]\n"
            )
        else:
            text += (
                f"Center xy ({x0_unit:.1f}, {y0_unit:.1f}) unit\n"
                f"Sigma xy ({sigma_x_unit:.1f}, {sigma_y_unit:.1f}) unit\n"
                f"Rotation {rot_angle:.1f}°\n\n"
                f"RF diameter {rf_diameter_unit:.1f} unit\n"
                f"RF area {rf_area_unit2:.1f} unit²\n"
            )
        # text += f"RF poly area {rf_poly_area_unit2:.1f} unit²\n"
        text += "\n"
        text += f"SNR (peak/std): {snr1:.2f}\n"
        text += f"SNR (gauss prj/resid): {snr2:.2f}\n"
        text += f"SNR (in/out): {snr3:.2f}\n"
        ax, im = utils.plot_sta(
            ax,
            spatial_sta,
            sta_analysis["EllipseCoor"],
            level_factor=level_factor,
            color="yellow",
            alpha=1,
            lw=line_width,
            linestyles="solid",
        )
        cax = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Signed STA: sign is the cell's polarity (bright = ON, dark = OFF), magnitude the
        # strength; normalised to +-1.
        cax.set_label("Spatial STA (norm.): +ON / -OFF", fontsize=fontsize_labels)
        cax.ax.tick_params(labelsize=fontsize_labels)
        ax.scatter(
            sta_analysis["Temporal_STA_coords"][0],
            sta_analysis["Temporal_STA_coords"][1],
            color=components_coords_color,
            marker="+",
            s=50,
            label="Temporal STA coords",
        )
        ax.legend(
            loc="lower right",
            fontsize=fontsize_labels,
            bbox_to_anchor=(-0.2, 0),
            frameon=False,
        )

        ax.set_title(title, fontsize=fontsize)
        if "Spatial_unit_size_um" in sta_analysis:
            pixel_size_um = sta_analysis["Spatial_unit_size_um"]
            xlabel = f"{nx} unit ({nx * pixel_size_um:.0f} µm)"
            ylabel = f"{ny} unit ({ny * pixel_size_um:.0f} µm)"
            utils.add_scalebar(
                ax,
                scalebar_size_um,
                pixel_size_um,
                scalebar_left_location,
                nx,
                ny,
                scale_bar_color,
                scale_bar_width,
            )
            text += f"\nScale bar: {scalebar_size_um} µm"
        else:
            xlabel = f"{nx} unit"
            ylabel = f"{ny} unit"
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.text(
            -0.25,
            1,
            text,
            transform=ax.transAxes,
            fontsize=fontsize_labels,
            va="top",
            ha="right",
        )
        next_ax += 1

        if add_spatial_mask:
            ax = axs[next_ax]
            if "Spatial_mask" in sta_analysis:
                spatial_mask = sta_analysis["Spatial_mask"]
                # Normalize the spatial mask to [0, 1] for visualization
                spatial_mask += np.abs(np.min(spatial_mask))
                spatial_mask /= np.max(spatial_mask)
                # Draw the ellipse fitted on the MASK. EllipseCoor is that mask fit for both
                # checkerboard and SWN now. (Older SWN results overwrote EllipseCoor with the
                # whitened-STA fit and kept the mask fit under EllipseCoor_raw — fall back to
                # it so those still display correctly.)
                mask_ellipse = sta_analysis.get(
                    "EllipseCoor_raw", sta_analysis["EllipseCoor"]
                )
                ax, im = utils.plot_sta(
                    ax,
                    spatial_mask,
                    mask_ellipse,
                    level_factor=level_factor,
                    color="yellow",
                    alpha=1,
                    lw=line_width,
                    linestyles="solid",
                    cmap="grey",
                    symmetric_colorbar=False,
                )
                cax = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Mask = how much each pixel varies over time (unsigned), normalised to [0, 1];
                # it locates the RF but carries no ON/OFF information.
                cax.set_label("Temporal variation (norm.)", fontsize=fontsize_labels)
                cax.ax.tick_params(labelsize=fontsize_labels)
                if "Spatial_unit_size_um" in sta_analysis:
                    pixel_size_um = sta_analysis["Spatial_unit_size_um"]
                    xlabel = f"{nx} unit ({nx * pixel_size_um:.0f} µm)"
                    ylabel = f"{ny} unit ({ny * pixel_size_um:.0f} µm)"
                    utils.add_scalebar(
                        ax,
                        scalebar_size_um,
                        pixel_size_um,
                        scalebar_left_location,
                        nx,
                        ny,
                        "chartreuse",  # mask is grey -> chartreuse; the STA uses scale_bar_color (black)
                        scale_bar_width,
                    )
                    text += f"\nScale bar: {scalebar_size_um} µm"
                else:
                    xlabel = f"{nx} unit"
                    ylabel = f"{ny} unit"
                ax.set_xlabel(xlabel, fontsize=fontsize)
                ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_title("Spatial mask", fontsize=fontsize)
            next_ax += 1

        # plot temporal sta
        ax = axs[next_ax]
        ax.set_title("Temporal STA", fontsize=fontsize)
        tsta_y = sta_analysis["Temporal"]
        peak_value = (
            tsta_y[sta_analysis["Cell_delay"]]
            if sta_analysis["Cell_delay"] is not None
            and not np.isnan(sta_analysis["Cell_delay"])
            else None
        )
        if (
            "TemporalTimeVector_s" in sta_analysis
            and "TemporalFreq_s" in sta_analysis
            and "Cell_delay_s" in sta_analysis
        ):
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
            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=fontsize_labels,
                va="top",
                ha="left",
            )
        else:
            ax.text(
                0.05,
                0.05,
                text,
                transform=ax.transAxes,
                fontsize=fontsize_labels,
                va="bottom",
                ha="left",
            )

        ax.plot(tsta_x, tsta_y, lw=line_width)
        if cell_delay is not None and not np.isnan(cell_delay):
            ax.axvline(
                cell_delay,
                color=components_coords_color,
                lw=line_width,
                ls="--",
                label="Spatial STA delay",
            )
            ax.legend(
                loc="lower left",
                fontsize=fontsize_labels,
                frameon=False,
                bbox_to_anchor=(1, 0),
            )
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel("STA amplitude", fontsize=fontsize)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        next_ax += 1

        if add_raster_plot:
            ax = axs[next_ax]
            ax.eventplot(rep_seq_data[cell_id]["spike_trains"])
            ax.set_title("Raster plot", fontsize=fontsize)
            ax.set_xlabel("Time (s)", fontsize=fontsize)
            ax.set_ylabel(
                f"{len(rep_seq_data[cell_id]['spike_trains'])} repetitions",
                fontsize=fontsize,
            )
            ax.set_aspect("auto")
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
