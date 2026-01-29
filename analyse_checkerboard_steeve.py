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
import temporary_utils
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
        params.output_directory, recording_number, "Checkerboard"
    )

    return (
        recording_number,
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

    print(f"\nCheckerboard Stats :")
    print(f"\t- {int(stim_onsets['duration']/params.fs/60)} min total duration")
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
    triggers_path = os.path.normpath(
        os.path.join(
            params.triggers_directory,
            f"{params.exp}_{params.checkerboard_name}_triggers.pkl",
        )
    )
    stim_onsets = utils.load_stim_onset_from_triggers_path(triggers_path, params, verbose=True)
    cells_id, checkerboard_spikes = utils.load_spike_times(
        params, params.checkerboard_name
    )
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


def compute_rasters(checkerboard_spikes, triggers, nb_repeats, stimulus_frequency):

    # initialiser raster output
    raster_data = {}

    # report status
    print('Computing rasters...')

    # loop over the spikes recorded during the checkerboard experiment
    # get the rasters on repeated sequence
    for (cell_id, spike_times) in tqdm(checkerboard_spikes.items()):
        raster_data[cell_id] = utils.extract_from_sequence(spike_times, triggers, nb_repeats, stim_frequency = stimulus_frequency)
    return raster_data


def plot_rasters(raster_data, cells_id, ploting:bool=True):

    # Plot all the rasters. Takes a few seconds.
    if ploting:
        size = int(math.sqrt(len(cells_id)))+1

        # setup subplots
        fig, axs = plt.subplots(nrows = size, ncols=size, figsize = (50,50))
        print('Ploting...')
        for i in tqdm(range(size**2)):
            ax = axs[i//size,i%size]
            if i < len(cells_id):
                ax.eventplot(raster_data[cells_id[i]]["spike_trains"])
                ax.set(title = "Cell {}".format(cells_id[i]),xlabel='Time in sec', ylabel='N Repetitions')
            else : ax.set_visible(False)

        # format and close
        plt.tight_layout()
        plt.show(block=False)
        plt.close('all')


def save_plots(raster_data, cells_id, recording_number:int, 
               check_directory:str, params:dict):
    """Create a folder path with the saved raster and 
    psths plots, a file per cell.
    
    Args:
        raster_data
        cells_id
        recording_number (int): 
        check_directory (str):
        params (dict):

    Returns:
    """
    # report status
    print("Saving rasters ...")
    
    # figure path
    fig_directory = os.path.normpath(os.path.join(
        check_directory, r'Rasters_figs'.format(recording_number)))
    
    # ensure path figure exists
    if not os.path.isdir(fig_directory): 
        os.makedirs(fig_directory)
    
    # loop over cells
    for cell_nb in tqdm(cells_id):

        # setup subplots
        fig, axs = plt.subplots(nrows = 2,ncols = 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,10))

        # add title
        plt.suptitle(f'Cell {cell_nb}')
        
        # plot raster
        ax_rast = axs[0]
        ax_rast.eventplot(raster_data[cell_nb]["spike_trains"])
        ax_rast.set(title = "Raster plot", ylabel='N Repetitions')

        # plot firing rate psth
        ax_psth = axs[1]
        width = (raster_data[cell_nb]["repeated_sequences_times"][0][0]/int(params.nb_frames_by_sequence/2))
        seq_lenght = raster_data[cell_nb]["repeated_sequences_times"][0][1]-raster_data[cell_nb]["repeated_sequences_times"][0][0]
        ax_psth.bar(np.linspace(0, seq_lenght, int(params.nb_frames_by_sequence/2)) + width/2, 
                    raster_data[cell_nb]["psth"], width=1.3*width)
        ax_psth.set(xlabel='Time in sec', ylabel='Firing rate (spikes/s)')

        # format figure
        plt.subplots_adjust(wspace=0, hspace=0)

        # save figure
        fig_file = os.path.join(fig_directory,f'Cell_{cell_nb}.png')
        plt.savefig(fig_file, dpi=fig.dpi)

        # clear and close figure
        plt.clf()
        plt.close()
    
    # save raster data
    np.save(os.path.join(check_directory,'Check_rasters_data'), raster_data)

def plot_all_cells_rasters(raster_data: dict, cells_id: list, plotting: bool = True):
    """
    Plot raster plots for all cells in a grid.

    Args:
        raster_data: Dictionary mapping cell IDs to raster data
        cells_id: List of cell IDs
        plotting: Whether to display plots (default True)
    """
    if not plotting:
        return

    size = int(math.sqrt(len(cells_id))) + 1
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(50, 50))

    print("Plotting...")
    for i in tqdm(range(size**2)):
        ax = axs[i // size, i % size]

        if i < len(cells_id):
            spike_trains = raster_data[cells_id[i]]["spike_trains"]
            temporary_utils.plot_single_raster(ax, spike_trains)
            ax.set(
                title=f"Cell {cells_id[i]}",
                xlabel="Time in sec",
                ylabel="N Repetitions",
            )
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.close("all")


def plot_one_cell_raster_and_psth(raster_data: dict, cell_nb, params: dict, show=False):
    """
    Interactively plot raster and PSTH for one user-selected cell.

    Args:
        raster_data: Dictionary with raster data
        checkerboard_spikes: Dictionary with spike data
        cells_id: List of available cell IDs
        params: Dictionary with experiment parameters
        show: Whether to display the plot (default False)
    """

    cell_nb = int(input("Select a cell: "))

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 10),
    )
    plt.suptitle(f"Cell {cell_nb}")
    temporary_utils.plot_single_raster(axs[0], raster_data[cell_nb]["spike_trains"])
    temporary_utils.plot_single_psth_from_raster_data(
        axs[1], raster_data, cell_nb, params
    )

    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show(block=False)
    plt.close(fig)
    return fig


def save_plots(
    raster_data: dict,
    cells_id: list,
    check_directory: str,
    params: dict,
):
    """
    Save raster and PSTH plots for all cells to files.

    Creates one PNG file per cell with raster plot and PSTH.

    Args:
        raster_data: Dictionary mapping cell IDs to raster data
        cells_id: List of cell IDs
        check_directory: Base directory for outputs
        params: Dictionary with experiment parameters
    """
    print("Saving rasters ...")

    fig_directory = os.path.normpath(os.path.join(check_directory, "Rasters_figs"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    for cell_nb in tqdm(cells_id):
        fig = plot_one_cell_raster_and_psth(raster_data, cell_nb, params, show=False)
        fig_file = os.path.join(fig_directory, f"Cell_{cell_nb}.png")
        plt.savefig(fig_file, dpi=fig.dpi)
        plt.clf()
        plt.close()

    np.save(os.path.join(check_directory, "Check_rasters_data"), raster_data)


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
