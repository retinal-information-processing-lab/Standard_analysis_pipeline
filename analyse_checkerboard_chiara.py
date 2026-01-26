"""Module of functions for the checkerboard analysis

boscarino.idv@gmail.com
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import utils



# ==============================================================================
# EXPERIMENTAL SETUP
# ==============================================================================

def prompt_user_for_recording(params: dict) -> int:
    """
    Display available recordings and prompt user to select one.
    
    Args:
        params: Dictionary containing experiment parameters with 'recording_names' key
        
    Returns:
        Selected recording number as integer
    """
    print('Which of the following is the checkerboard: ')
    for num, rec in enumerate(params.recording_names):
        print(f'\t{num} --> {rec}')
    
    recording_number = int(input('Checkerboard number : '))
    params.checkerboard_name = params.recording_names[recording_number]
    print(f'Selected recording: {params.checkerboard_name}\n')
    
    return recording_number


def prompt_user_for_stimulus_params() -> tuple[int, int, int]:
    """
    Prompt user for stimulus parameters.
    
    Returns:
        Tuple of (stimulus_frequency, nb_checks_x, nb_checks_y)
    """
    stimulus_frequency = int(input("Select stimulus frequency (usually 30Hz) : "))
    nb_checks_x = int(input("Select number of checks on x (usually 40 for fine checkerboard) : "))
    nb_checks_y = int(input("Select number of checks on y (usually 40 for fine checkerboard) : "))
    
    return stimulus_frequency, nb_checks_x, nb_checks_y


def create_analysis_directory(params: dict, recording_number: int) -> str:
    """
    Create directory for checkerboard analysis output.
    
    Args:
        params: Dictionary containing 'output_directory' key
        recording_number: Recording number for directory naming
        
    Returns:
        Path to created directory
    """
    check_directory = os.path.normpath(
        os.path.join(params.output_directory, f'Checkerboard_Analysis_rec_{recording_number}')
    )
    
    if not os.path.isdir(check_directory):
        os.makedirs(check_directory)
    
    return check_directory


def get_inputs(params: dict) -> tuple[int, int, int, int, str]:
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
    recording_number = prompt_user_for_recording(params)
    stimulus_frequency, nb_checks_x, nb_checks_y = prompt_user_for_stimulus_params()
    check_directory = create_analysis_directory(params, recording_number)
    
    return recording_number, stimulus_frequency, nb_checks_x, nb_checks_y, check_directory


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_triggers(params: dict) -> tuple[np.ndarray, dict]:
    """
    Load trigger data from saved file.
    
    Args:
        params: Dictionary with 'triggers_directory', 'exp', and 'checkerboard_name'
        
    Returns:
        Tuple of (triggers array, triggers_data dict)
    """
    triggers_file = os.path.normpath(os.path.join(
        params.triggers_directory,
        f"{params.exp}_{params.checkerboard_name}_triggers.pkl"
    ))
    triggers_data = utils.load_obj(triggers_file)
    triggers = triggers_data['indices'] / params.fs
    
    return triggers, triggers_data


def load_spike_data(params: dict) -> tuple[dict, list]:
    """
    Load spike data for checkerboard recording.
    
    Args:
        params: Dictionary with experiment parameters
        
    Returns:
        Tuple of (checkerboard_spikes dict, cells_id list)
    """
    neurons_file = os.path.normpath(os.path.join(
        params.output_directory,
        f'{params.exp}_fullexp_neurons_data.pkl'
    ))
    all_recs_spikes = utils.load_obj(neurons_file)
    checkerboard_spikes = utils.get_recording_spikes(params.checkerboard_name, all_recs_spikes)
    cells_id = list(checkerboard_spikes.keys())
    
    return checkerboard_spikes, cells_id


def calculate_experiment_stats(triggers_data: dict, triggers: np.ndarray, params: dict, 
                               stimulus_frequency: int) -> tuple[int, int]:
    """
    Calculate and display experiment statistics.
    
    Args:
        triggers_data: Dictionary containing trigger duration
        triggers: Array of trigger times
        params: Dictionary with 'fs' and 'nb_frames_by_sequence'
        stimulus_frequency: Stimulus frequency in Hz
        
    Returns:
        Tuple of (nb_repeats, duration_sequence)
    """
    nb_repeats = int(len(triggers) / params.nb_frames_by_sequence)
    duration_sequence = int(params.nb_frames_by_sequence / stimulus_frequency)
    
    print(f"\nCheckerboard Stats :")
    print(f"\t- {int(triggers_data['duration']/params.fs/60)} min total duration")
    print(f"\t- {len(triggers)} triggers")
    print(f"\t- {nb_repeats} complete sequences")
    print(f"\t- {duration_sequence} seconds per sequence\n")
    
    return nb_repeats, duration_sequence


def load_or_create_stimulus(nb_repeats: int, nb_checks_x: int, nb_checks_y: int,
                            check_directory: str, params: dict) -> np.ndarray:
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
    stimulus_path = os.path.normpath(os.path.join(
        check_directory,
        f"checkerboard_{nb_checks_x}x{nb_checks_y}checks_{nb_frames}frames.npy"
    ))
    
    if os.path.isfile(stimulus_path):
        print(f"Stimulus file exists. Loaded from:\t {stimulus_path}")
        checkerboard = np.load(stimulus_path)
    else:
        print("Reconstructing the stimulus...")
        checkerboard = checkerboard_from_binary(
            nb_frames, nb_checks_x, nb_checks_y,
            checkerboard_file=stimulus_path,
            binary_source_path=params.binary_source_path
        )
    
    return checkerboard


def load_checkerboard_experiment_data(params: dict, check_directory: str, 
                                     nb_checks_x: int, nb_checks_y: int, 
                                     stimulus_frequency: int) -> tuple:
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
    triggers, triggers_data = load_triggers(params)
    checkerboard_spikes, cells_id = load_spike_data(params)
    nb_repeats, _ = calculate_experiment_stats(triggers_data, triggers, params, stimulus_frequency)
    checkerboard = load_or_create_stimulus(nb_repeats, nb_checks_x, nb_checks_y, check_directory, params)
    
    print(f'Total : {len(checkerboard_spikes.keys())} neurons loaded\n\nClusters id :\n{cells_id}\n')
    
    return checkerboard_spikes, triggers, nb_repeats, cells_id, checkerboard


# ==============================================================================
# RASTER ANALYSIS
# ==============================================================================

def compute_rasters(checkerboard_spikes: dict, triggers: np.ndarray, 
                   nb_repeats: int, stimulus_frequency: int) -> dict:
    """
    Compute raster data for all cells.
    
    Args:
        checkerboard_spikes: Dict mapping cell IDs to spike times
        triggers: Array of trigger times
        nb_repeats: Number of stimulus repetitions
        stimulus_frequency: Stimulus frequency in Hz
        
    Returns:
        Dictionary mapping cell IDs to raster data
    """
    print('Computing rasters...')
    raster_data = {}
    
    for cell_id, spike_times in tqdm(checkerboard_spikes.items()):
        raster_data[cell_id] = utils.extract_from_sequence(
            spike_times, triggers, nb_repeats, stim_frequency=stimulus_frequency
        )
    
    return raster_data


def create_raster_grid(cells_id: list) -> tuple:
    """
    Create figure and axes grid for raster plots.
    
    Args:
        cells_id: List of cell IDs
        
    Returns:
        Tuple of (fig, axs, size)
    """
    size = int(math.sqrt(len(cells_id))) + 1
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(50, 50))
    return fig, axs, size


def plot_rasters(raster_data: dict, cells_id: list, plotting: bool = True):
    """
    Plot raster plots for all cells in a grid.
    
    Args:
        raster_data: Dictionary mapping cell IDs to raster data
        cells_id: List of cell IDs
        plotting: Whether to display plots (default True)
    """
    if not plotting:
        return
    
    fig, axs, size = create_raster_grid(cells_id)
    
    print('Plotting...')
    for i in tqdm(range(size ** 2)):
        ax = axs[i // size, i % size]
        
        if i < len(cells_id):
            ax.eventplot(raster_data[cells_id[i]]["spike_trains"])
            ax.set(title=f"Cell {cells_id[i]}", xlabel='Time in sec', ylabel='N Repetitions')
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.close('all')


def create_single_cell_raster_plot(raster_data: dict, cell_nb: int, params: dict) -> tuple:
    """
    Create figure with raster and PSTH for a single cell.
    
    Args:
        raster_data: Dictionary with raster data
        cell_nb: Cell number to plot
        params: Dictionary with 'nb_frames_by_sequence'
        
    Returns:
        Tuple of (fig, axs)
    """
    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(10, 10)
    )
    plt.suptitle(f'Cell {cell_nb}')
    
    return fig, axs


def plot_single_raster(ax, raster_data: dict, cell_nb: int):
    """
    Plot raster for a single cell.
    
    Args:
        ax: Matplotlib axis
        raster_data: Dictionary with raster data
        cell_nb: Cell number to plot
    """
    ax.eventplot(raster_data[cell_nb]["spike_trains"])
    ax.set(title="Raster plot", ylabel='N Repetitions')


def plot_single_psth(ax, raster_data: dict, cell_nb: int, params: dict):
    """
    Plot PSTH for a single cell.
    
    Args:
        ax: Matplotlib axis
        raster_data: Dictionary with raster data
        cell_nb: Cell number to plot
        params: Dictionary with 'nb_frames_by_sequence'
    """
    width = (raster_data[cell_nb]["repeated_sequences_times"][0][0] / 
             int(params.nb_frames_by_sequence / 2))
    seq_length = (raster_data[cell_nb]["repeated_sequences_times"][0][1] - 
                  raster_data[cell_nb]["repeated_sequences_times"][0][0])
    
    x_vals = np.linspace(0, seq_length, int(params.nb_frames_by_sequence / 2)) + width / 2
    ax.bar(x_vals, raster_data[cell_nb]["psth"], width=1.3 * width)
    ax.set(xlabel='Time in sec', ylabel='Firing rate (spikes/s)')


def save_plots(raster_data: dict, cells_id: list, recording_number: int, 
              check_directory: str, params: dict):
    """
    Save raster and PSTH plots for all cells to files.
    
    Creates one PNG file per cell with raster plot and PSTH.
    
    Args:
        raster_data: Dictionary mapping cell IDs to raster data
        cells_id: List of cell IDs
        recording_number: Recording number for directory naming
        check_directory: Base directory for outputs
        params: Dictionary with experiment parameters
    """
    print("Saving rasters ...")
    
    fig_directory = os.path.normpath(os.path.join(check_directory, 'Rasters_figs'))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)
    
    for cell_nb in tqdm(cells_id):
        fig, axs = create_single_cell_raster_plot(raster_data, cell_nb, params)
        plot_single_raster(axs[0], raster_data, cell_nb)
        plot_single_psth(axs[1], raster_data, cell_nb, params)
        
        plt.subplots_adjust(wspace=0, hspace=0)
        fig_file = os.path.join(fig_directory, f'Cell_{cell_nb}.png')
        plt.savefig(fig_file, dpi=fig.dpi)
        plt.clf()
        plt.close()
    
    np.save(os.path.join(check_directory, 'Check_rasters_data'), raster_data)


def plot_one_cell_raster_and_psth(raster_data: dict, checkerboard_spikes: dict, 
                                 cells_id: list, params: dict):
    """
    Interactively plot raster and PSTH for one user-selected cell.
    
    Args:
        raster_data: Dictionary with raster data
        checkerboard_spikes: Dictionary with spike data
        cells_id: List of available cell IDs
        params: Dictionary with experiment parameters
    """
    print(f'Total : {len(checkerboard_spikes.keys())} neurons found\n\nClusters id :\n{cells_id}\n')
    
    cell_nb = int(input("Select a cell: "))
    
    fig, axs = create_single_cell_raster_plot(raster_data, cell_nb, params)
    plot_single_raster(axs[0], raster_data, cell_nb)
    plot_single_psth(axs[1], raster_data, cell_nb, params)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show(block=False)
    plt.close(fig)


# ==============================================================================
# SPIKE-TRIGGERED AVERAGE ANALYSIS
# ==============================================================================

def compute_spike_triggered_average(checkerboard_spikes: dict, checkerboard: np.ndarray, 
                                   triggers: np.ndarray, nb_repeats: int, 
                                   stimulus_frequency: int, check_directory: str,
                                   sequence_portion: tuple = (0, 0.5),
                                   sta_data_filename: str = 'sta_data_3D.pkl') -> tuple:
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
    print('Computing STAs...')
    sta_data = {}
    
    for cell_id, spike_times in tqdm(checkerboard_spikes.items()):
        sta_data[cell_id] = utils.extract_from_sequence(
            spike_times, triggers, nb_repeats, stimulus_frequency,
            sequence_portion=sequence_portion
        )
        sta_3D = utils.compute_3D_sta(sta_data[cell_id], checkerboard, 
                                      stimulus_frequency, cluster_id=cell_id)
        sta_data[cell_id]['sta_3D'] = sta_3D
    
    sta_data_file = os.path.normpath(os.path.join(check_directory, sta_data_filename))
    utils.save_obj(sta_data, sta_data_file)
    
    return sta_data_file, sta_data


def plot_one_cell_3D_spike_triggered_average(sta_data: dict, checkerboard_spikes: dict, 
                                            cells_id: list):
    """
    Interactively plot 3D STA for one user-selected cell.
    
    Displays all temporal frames of the STA in a grid.
    
    Args:
        sta_data: Dictionary with STA data
        checkerboard_spikes: Dictionary with spike data
        cells_id: List of available cell IDs
    """
    print(f'Total : {len(checkerboard_spikes.keys())} neurons found\n\nClusters id :\n{cells_id}\n')
    
    cell_id = int(input("Select a cell: "))
    
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(8, 5, figure=fig)
    
    for i in range(40):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.imshow(sta_data[cell_id]['sta_3D'][i])
    
    plt.show(block=False)
    plt.close(fig)


# ==============================================================================
# ELLIPSE FITTING
# ==============================================================================

def fit_ellipse_single_cell(cell_id: int, sta_3D: np.ndarray, 
                           checkerboard: np.ndarray, method: str) -> dict:
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
    if np.max(np.abs(sta_3D)) == 0:
        print(f'No ellipse fit on cell {cell_id}')
        return {
            'Spatial': np.zeros(checkerboard[0].shape),
            'Temporal': np.zeros(checkerboard.shape[0]),
            'EllipseCoor': np.asarray([0, 0, 0, 0.001, 0.001, 0]),
            'Cell_delay': np.nan
        }
    
    if method == 'tom':
        return utils.analyse_sta_tom(sta_3D, cell_id)
    elif method == 'matias':
        return utils.analyse_sta_matias(sta_3D, cell_id)
    elif method == 'gab':
        return utils.analyse_sta_gab(sta_3D, cell_id)
    elif method == 'basic':
        return utils.analyse_sta(sta_3D, cell_id)


def fit_ellipse_to_spike_triggered_average(checkerboard: np.ndarray, 
                                          check_directory: str,
                                          method: str = "tom",
                                          fitted_sta_filename: str = 'sta_data_3D_fitted.pkl'):
    """
    Fit ellipses to STAs for all cells.
    
    Fits 2D Gaussian ellipses to spatial receptive fields from spike-triggered
    averages. Supports multiple fitting methods.
    
    Args:
        checkerboard: Stimulus array for reference dimensions
        check_directory: Directory containing and for saving STA data
        method: Fitting method - 'tom', 'matias', 'gab', or 'basic'
        fitted_sta_filename: Output filename for fitted data
    """
    sta_data_file = os.path.normpath(os.path.join(check_directory, 'sta_data_3D.pkl'))
    sta_data = utils.load_obj(sta_data_file)
    
    for cell_id in tqdm(sta_data):
        sta_3D = sta_data[cell_id]['sta_3D']
        
        result_key = {
            'tom': 'center_analyse',
            'matias': 'center_analyse',
            'gab': 'surround_analyse',
            'basic': 'analyse_sta'
        }.get(method, 'center_analyse')
        
        sta_data[cell_id][result_key] = fit_ellipse_single_cell(
            cell_id, sta_3D, checkerboard, method
        )
    
    fitted_file = os.path.normpath(os.path.join(check_directory, fitted_sta_filename))
    utils.save_obj(sta_data, fitted_file)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_sta_figure_directory(check_directory: str) -> str:
    """
    Create directory for STA figures.
    
    Args:
        check_directory: Base analysis directory
        
    Returns:
        Path to figures directory
    """
    fig_directory = os.path.normpath(os.path.join(check_directory, 'Stas_figs'))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)
    return fig_directory


def plot_sta_fitted_with_ellipse(cells_id: list, cells_to_plot: list, 
                                 check_directory: str):
    """
    Plot fitted STAs with ellipses for all cells.
    
    Creates and saves plots showing spatial STA and fitted ellipse for each cell.
    Only displays plots for cells specified in cells_to_plot.
    
    Args:
        cells_id: List of all cell IDs to process
        cells_to_plot: List of cell IDs to display interactively
        check_directory: Directory containing fitted STA data
    """
    fig_directory = create_sta_figure_directory(check_directory)
    sta_data = np.load(os.path.join(check_directory, 'sta_data_3D_fitted.pkl'), 
                      allow_pickle=True)
    
    for cell_id in tqdm(cells_id):
        sta = sta_data[cell_id]["center_analyse"]
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
        plt.suptitle(f'Cell {cell_id}')
        
        axs[0].imshow(sta['Spatial'])
        axs[0].set(title="Spatial STA")
        
        axs[1].set(title="Fitted Ellipse")
        try:
            axs[1] = plot_sta(axs[1], sta['Spatial'], sta['EllipseCoor'])
        except:
            axs[1].imshow(sta_data[cell_id]["center_analyse"]['Spatial'])
        
        fig_file = os.path.join(fig_directory, f'Cell_{cell_id}.png')
        plt.savefig(fig_file, dpi=fig.dpi)
        
        if cell_id in cells_to_plot:
            plt.show()
        
        plt.close(fig)


def plot_sta_fitted_with_ellipse_by_tom(raster_data: dict, cells_id: list, 
                                       cells_to_plot: list, check_directory: str):
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
    fig_directory = create_sta_figure_directory(check_directory)
    sta_data = np.load(os.path.join(check_directory, 'sta_data_3D_fitted.pkl'), 
                      allow_pickle=True)
    
    for cell_id in tqdm(cells_id):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
        plt.suptitle(f'Cell {cell_id}')
        
        sta = sta_data[cell_id]["center_analyse"]
        
        # Raster
        axs[0].eventplot(raster_data[cell_id]["spike_trains"])
        axs[0].set(title="Raster plot", ylabel='N Repetitions')
        
        # Fitted ellipse
        axs[1].set(title="Fitted Ellipse")
        try:
            axs[1] = utils.plot_sta_tom(axs[1], sta['Spatial'], sta['EllipseCoor'])
        except:
            axs[1].imshow(sta_data[cell_id]["center_analyse"]['Spatial'])
        
        # Temporal
        axs[2].set(title="Temporal STA")
        axs[2].plot(sta['Temporal'])
        axs[2].set_ylim([-1, 1])
        
        fig_file = os.path.join(fig_directory, f'Cell_{cell_id}.png')
        plt.savefig(fig_file, dpi=fig.dpi)
        
        if cell_id in cells_to_plot:
            plt.show()
        
        plt.clf()
        plt.close(fig)



def main():

    import params
    # ask user to input the experiment parameters
    # e.g., for the test dataset enter inputs: 0, 30, 42, 42
    recording_number, stimulus_frequency, nb_checks_x, nb_checks_y, check_directory = get_inputs(params)

    # load checkerboard experiment data
    checkerboard_spikes, triggers, nb_repeats, cells_id, checkerboard = load_checkerboard_experiment_data(params, check_directory, nb_checks_x, nb_checks_y, stimulus_frequency) 
    
if __name__ == "__main__":
    main()