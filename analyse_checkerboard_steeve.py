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

# import custom packages
import utils


# experimental design ---------------------------


def prompt_user_for_checkerboard_params() -> tuple[int, int, int]:
    """
    Prompt user for stimulus parameters.

    Returns:
        Tuple of (stimulus_frequency, nb_checks_x, nb_checks_y)
    """
    stimulus_frequency = int(input("Select stimulus frequency (usually 30Hz) : "))
    nb_checks_x = int(
        input("Select number of checks on x (usually 40 for fine checkerboard) : ")
    )
    nb_checks_y = int(
        input("Select number of checks on y (usually 40 for fine checkerboard) : ")
    )

    return stimulus_frequency, nb_checks_x, nb_checks_y


def get_all_inputs_for_checkerboard_analysis(
    params: dict,
) -> tuple[int, int, int, int, str]:
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
            - check_directory: Path to analysis output directory
    """
    recording_number, recording_name = utils.prompt_user_for_recording(
        params, "checkerboard"
    )
    stimulus_frequency, nb_checks_x, nb_checks_y = prompt_user_for_checkerboard_params()
    check_directory = utils.create_analysis_directory(
        params, recording_number, "Checkerboard"
    )

    return (
        recording_number,
        recording_name,
        stimulus_frequency,
        nb_checks_x,
        nb_checks_y,
        check_directory,
    )


def calculate_checkerboard_experiment_stats(
    stim_onsets: dict, triggers: np.ndarray, params: dict, stimulus_frequency: int
) -> tuple[int, int]:
    """
    Calculate and display experiment statistics.

    Args:
        stim_onsets: Dictionary containing trigger duration
        triggers: Array of trigger times
        params: Dictionary with 'fs' and 'nb_frames_by_sequence'
        stimulus_frequency: Stimulus frequency in Hz

    Returns:
        Tuple of (nb_repeats, duration_sequence)
    """
    nb_repeats = int(len(triggers) / params.nb_frames_by_sequence)
    duration_sequence = int(params.nb_frames_by_sequence / stimulus_frequency)

    print("\nCheckerboard Stats :")
    print(f"\t- {int(stim_onsets[-1] / 60)} min total duration")
    print(f"\t- {len(triggers)} triggers")
    print(f"\t- {nb_repeats} complete sequences")
    print(f"\t- {duration_sequence} seconds per sequence\n")

    return nb_repeats, duration_sequence


def load_or_create_checkerboard_stimulus(
    nb_repeats: int,
    nb_checks_x: int,
    nb_checks_y: int,
    check_directory: str,
    params: dict,
) -> np.ndarray:
    """
    Load existing stimulus array or create new one.

    Args:
        nb_repeats: Number of stimulus repetitions
        nb_checks_x: Number of checks in x dimension
        nb_checks_y: Number of checks in y dimension
        check_directory: Directory for stimulus file
        params: Dictionary with 'nb_frames_by_sequence' and 'binary_source_path'

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
    params: dict,
    check_directory: str,
    checkerboard_name: str,
    nb_checks_x: int,
    nb_checks_y: int,
    stimulus_frequency: int,
) -> tuple:
    """
    Load and process all checkerboard experiment data.

    Loads triggers, spikes, and stimulus data. Calculates experiment statistics.

    Args:
        params: Dictionary with experiment parameters
        check_directory: Directory for analysis outputs
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
        stim_onsets, stim_onsets, params, stimulus_frequency
    )
    checkerboard = load_or_create_checkerboard_stimulus(
        nb_repeats, nb_checks_x, nb_checks_y, check_directory, params
    )

    print(
        f"Total : {len(checkerboard_spikes.keys())} neurons loaded\n\nClusters id :\n{cells_id}\n"
    )

    return checkerboard_spikes, stim_onsets, nb_repeats, cells_id, checkerboard


# rasters and psths ---------------

def extract_all_cell_responses_to_repeated_sequences(
        checkerboard_spikes: dict, 
        triggers: np.ndarray, 
        nb_repeats: int, 
        stimulus_frequency: int,
        cells_id: list) -> dict:
    
    """
    Extract responses to repeated stimulus sequences for all cells.

    Args:
        checkerboard_spikes: Dict mapping cell IDs to spike times {cell_id: np.array of spike times}
        triggers: Array of trigger times (n_triggers,)
        nb_repeats: Number of complete stimulus sequences
        stimulus_frequency: Stimulus frequency in Hz

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
            spike_times, triggers, nb_repeats, stim_frequency=stimulus_frequency
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
    cells_id: list, 
    check_directory: str, 
    title: str = "Response to repeated sequence",
    fontsize: int = 14,
    save_figures: bool = True,
    show_figures: bool = False
):
    """
    Generate single cell raster plots (one figure per cell showing raster+psth) 
    and save them in a new folder: "Rasters_figs" in the check_directory.

    Args:
        rep_seq_data: Dict containing extracted responses for each cell (from extract_all_cell_responses_to_repeated_sequences)
        cells_id: List of cell IDs to plot
        check_directory: path to the directory where to generate the "Rasters_figs" folder and save the figures
        params: dict
    Returns:
        None
    """

    # figure path
    fig_directory = os.path.normpath(os.path.join(check_directory, r"Rasters_figs"))
    # ensure path figure exists
    if not os.path.isdir(fig_directory): 
        os.makedirs(fig_directory)

    # loop over cells
    for cell_nb in tqdm(cells_id, desc="Plotting and saving rasters for each cell"):
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
            seq_lenght=rep_seq_data[cell_nb]["repeated_sequences_times"][0][1]
                        - rep_seq_data[cell_nb]["repeated_sequences_times"][0][0],
            title=title,
            fontsize=fontsize,
        )

        # format figure
        plt.subplots_adjust(wspace=0, hspace=0)

        # save figure
        fig_file = os.path.join(fig_directory, f"Cell_{cell_nb}.png")
        if save_figures: 
            plt.savefig(fig_file, dpi=fig.dpi)
        if show_figures: 
            plt.show(block=False)
            print(f"fig_file : {fig_file}")

        # clear and close figure
        plt.clf()
        plt.close()

    return


# spike triggered averages ---------------


def compute_spike_triggered_average(
    checkerboard_spikes: dict,
    checkerboard: np.ndarray,
    triggers: np.ndarray,
    nb_repeats: int,
    stimulus_frequency: int,
    check_directory: str,
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
        )

        # Compute sta of the cell
        sta_3D = utils.compute_3D_sta(
            sta_data[cell_id], checkerboard, stimulus_frequency, cluster_id=cell_id
        )

        # Adding data to the notebook dictionnary
        sta_data[cell_id]["sta_3D"] = sta_3D

    # save
    sta_data_file = os.path.normpath(os.path.join(check_directory, sta_data_filename))
    utils.save_obj(sta_data, sta_data_file)
    return sta_data_file, sta_data


def plot_one_cell_3D_spike_triggered_average(
    sta_data: dict, checkerboard_spikes: dict, cells_id: list
):
    """
    Interactively plot 3D STA for one user-selected cell.

    Displays all temporal frames of the STA in a grid.

    Args:
        sta_data: Dictionary with STA data
        checkerboard_spikes: Dictionary with spike data
        cells_id: List of available cell IDs
    """
    # report number of cells
    print(
        "Total : {} neurons found \n\nClusters id :\n{}\n".format(
            len(checkerboard_spikes.keys()), cells_id
        )
    )

    # ask user to input a cell
    cell_id = int(input("Select a cell: "))

    # plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(8, 5, figure=fig)
    for i in range(40):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.imshow(sta_data[cell_id]["sta_3D"][i])
    plt.show(block=False)
    plt.close(fig)


def fit_ellipse_to_spike_triggered_average(
    cell_id: int, sta_3D: np.ndarray, checkerboard: np.ndarray, method: str
) -> dict:
    """
    Fit ellipse to STA for a single cell.

    Args:
        cell_id: Cell identifier
        sta_3D: 3D spike-triggered average
        checkerboard: Stimulus array
        method: Fitting method ('tom', 'matias', 'gab', or 'basic')

    Returns:
        Dictionary with fitted parameters or empty arrays if fitting failed
    """
    # File loading the 3D sta dictionnary saved above in this notebook
    check_directory = utils.find_analysis_directory("Checkerboard")
    sta_data_file = os.path.normpath(os.path.join(check_directory, "sta_data_3D.pkl"))
    sta_data = utils.load_obj(sta_data_file)

    # loop over cells
    for cell_id in tqdm(sta_data):
        # get 3D sta
        sta_3D = sta_data[cell_id]["sta_3D"]

        if np.max(np.abs(sta_3D)) == 0:
            print(f"No ellipse fit on cell {cell_id}")
            sta_data[cell_id]["center_analyse"] = {
                "Spatial": np.zeros(checkerboard[0].shape),
                "Temporal": np.zeros(checkerboard.shape[0]),
                "EllipseCoor": np.asarray([0, 0, 0, 0.001, 0.001, 0]),
                "Cell_delay": np.nan,
            }
            sta_data[cell_id]["surround_analyse"] = {
                "Spatial": np.zeros(checkerboard[0].shape),
                "Temporal": np.zeros(checkerboard.shape[0]),
                "EllipseCoor": np.asarray([0, 0, 0, 0.001, 0.001, 0]),
                "Cell_delay": np.nan,
            }
            sta_data[cell_id]["analyse_sta"] = {
                "Spatial": np.zeros(checkerboard[0].shape),
                "Temporal": np.zeros(checkerboard.shape[0]),
                "EllipseCoor": np.asarray([0, 0, 0, 0.001, 0.001, 0]),
                "Cell_delay": np.nan,
            }

            continue

        # Perform the fitting for the current cell using one of the bellow method (default "analyse_sta_matias")
        if method == "tom":
            # New fitting of ellipse, more performant
            sta_data[cell_id]["center_analyse"] = utils.analyse_sta_tom(sta_3D, cell_id)
        elif method == "matias":
            sta_data[cell_id]["center_analyse"] = utils.analyse_sta_matias(
                sta_3D, cell_id
            )
        elif method == "gab":
            sta_data[cell_id]["surround_analyse"] = utils.analyse_sta_gab(
                sta_3D, cell_id
            )
        elif method == "basic":
            sta_data[cell_id]["analyse_sta"] = utils.analyse_sta(sta_3D, cell_id)

    fitted_sta_filename = "sta_data_3D_fitted.pkl"
    # ouput file of all fitted data
    fitted_file = os.path.normpath(os.path.join(check_directory, fitted_sta_filename))

    # save file
    utils.save_obj(sta_data, fitted_file)


def plot_sta_fitted_with_ellipse(
    cells_id: list, cells_to_plot: list, check_directory: str
):
    """
    Plot fitted STAs with ellipses for all cells.

    Creates and saves plots showing spatial STA and fitted ellipse for each cell.
    Only displays plots for cells specified in cells_to_plot.

    Args:
        cells_id: List of all cell IDs to process
        cells_to_plot: List of cell IDs to display interactively
        check_directory: Directory containing fitted STA data
    """

    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory, r"Stas_figs"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    # load sta
    sta_data = np.load(
        os.path.join(check_directory, "sta_data_3D_fitted.pkl"), allow_pickle=True
    )

    # loop over all cells
    for cell_id in tqdm(cells_id[0:]):
        # get cell sta data
        sta = sta_data[cell_id]["center_analyse"]

        # setup subplots
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        # add title
        plt.suptitle(f"Cell {cell_id}")

        # plot spatial sta
        ax = axs[0]
        ax.imshow(sta["Spatial"])
        ax.set(title="Spatial STA")

        # plot fitted ellipse
        ax = axs[1]
        ax.set(title="Fitted Ellipse")

        try:
            ax = utils.plot_sta(ax, sta["Spatial"], sta["EllipseCoor"])
        except:
            ax.imshow(sta_data[cell_id]["center_analyse"]["Spatial"])

        fig_file = os.path.join(fig_directory, f"Cell_{cell_id}.png")

        # save figure
        plt.savefig(fig_file, dpi=fig.dpi)

        # plot selected cells only
        if cell_id in cells_to_plot:
            plt.show()

        # close figure to free memory
        plt.close(fig)


def plot_sta_fitted_with_ellipse_by_tom(
    raster_data: dict, cells_id: list, cells_to_plot: list, check_directory: str
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

        fig_file = os.path.join(fig_directory, f"Cell_{cell_id}.png")
        plt.savefig(fig_file, dpi=fig.dpi)

        # plot selected cells only
        if cell_id in cells_to_plot:
            plt.show()

        # close figure to free memory
        plt.clf()
        plt.close(fig)
