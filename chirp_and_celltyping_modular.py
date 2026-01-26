import numpy as np
import pickle
import os
import glob
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
import params
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
import itertools
import time
from collections import defaultdict
from math import *
import utils


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_all_inputs_for_chirp_analysis(params: dict, old: bool):
    """Load trigger times and spike data, then select chirp stimulus type.
    
    NOTE : 'old' needs to be reviewed

    Args:
        params (dict): Experiment parameters from params.py containing:
            - exp (str): Experiment name
            - recording_names (list): List of recording names
            - output_directory (str): Path to analysis output directory
            - triggers_directory (str): Path to trigger files
            - fs (float): Sampling rate of the MEA in Hz
        old (bool): If True, use old 2p room chirp. If False, use new 50Hz chirp.

    Returns:
        tuple: Contains:
            - cells (list[np.uint32]): List of neuron cluster IDs
            - spike_times (list[np.ndarray]): Spike times for each neuron
            - spike_trains (dict): Complete spike train data for all neurons
            - trig_data (dict): Trigger timing information
            - stim_onsets (np.ndarray): Stimulus onset times in seconds
            - check_directory (str): Path to checkerboard analysis directory
            - CT_directory (str): Path to cell typing output directory
            - old (bool): Chirp type flag (passed through)

    Note:
        Prompts user to select recording number from available recordings.
        Creates cell typing directory if it doesn't exist.
    """
    recording_names = params.recording_names
    output_directory = params.output_directory

    # Prompt user to select recording
    recording_number, rec = utils.prompt_user_for_recording(params, "chirp recording")
    print(f"\nSelected recording : {rec} \n")
    
    # Create cell typing directory
    CT_directory = utils.create_analysis_directory(params, recording_number, "CellTyping")
    
    check_directory = utils.find_Analysis_Directory(dir_type="Checkerboard")

    # Load triggers
    stim_onsets, trig_data = utils.load_triggers(params, rec)
    
    
    # Load spike trains
    cells, spike_times = utils.load_spike_trains(params, rec)
        
    print(f'Total : {len(spike_times)} neurons loaded \n\nClusters id :\n{cells}\n')

    return cells, spike_times, spike_times, trig_data, stim_onsets, check_directory, CT_directory, old

# =============================================================================
# CHIRP RESPONSE COMPUTATION
# =============================================================================

def get_chirp_parameters(old: bool):
    """Get chirp stimulus parameters based on stimulus type.

    Args:
        old (bool): If True, use old chirp parameters. If False, use new chirp parameters.

    Returns:
        tuple: Contains:
            - nb_repetitions (int): Number of stimulus repetitions
            - n_bins (int): Number of time bins for PSTH
            - rep_length (float): Length of each repetition in seconds
            - time_bin (float): Duration of each time bin in seconds
    """
    if old:
        nb_repetitions = 30
        n_bins = 625
        rep_length = 25
    else:
        nb_repetitions = 20
        n_bins = 800
        rep_length = 32
    
    time_bin = rep_length / n_bins
    
    return nb_repetitions, n_bins, rep_length, time_bin


def extract_repeated_sequence_times(stim_onsets: np.ndarray, nb_repetitions: int, old: bool):
    """Extract start and end times for each stimulus repetition.

    Args:
        stim_onsets (np.ndarray): Array of stimulus onset times in seconds
        nb_repetitions (int): Number of stimulus repetitions
        old (bool): If True, use old chirp indexing. If False, use new chirp indexing.

    Returns:
        list[list[float]]: List of [start_time, end_time] pairs for each repetition
    """
    repeated_sequences_times = []
    
    for i in range(nb_repetitions):
        if old:
            times = stim_onsets[i*999 : 999*(i+1)]
        else:
            times = stim_onsets[i*1600+151 : 151+1600*(i+1)]
        repeated_sequences_times.append([times[0], times[-1]])
    
    return repeated_sequences_times


def align_spike_trains(spike_times: np.ndarray, repeated_sequences_times: list):
    """Align spike trains to stimulus onset for each repetition.

    Args:
        spike_times (np.ndarray): Array of spike times in seconds
        repeated_sequences_times (list[list[float]]): List of [start, end] times for each repetition

    Returns:
        list[np.ndarray]: List of aligned spike trains, one per repetition
    """
    spike_trains = []
    
    for i, (start_time, end_time) in enumerate(repeated_sequences_times):
        # Restrict to current repetition window
        spike_train = restrict_array(spike_times, start_time, end_time)
        # Align to repetition start
        spike_train = spike_train - start_time
        spike_trains.append(spike_train)
    
    return spike_trains


def compute_binned_responses(spike_trains: list, nb_repetitions: int, n_bins: int, 
                             n_bins_small: int, rep_length: float):
    """Bin spike trains at multiple time scales.

    Args:
        spike_trains (list[np.ndarray]): Aligned spike trains for each repetition
        nb_repetitions (int): Number of stimulus repetitions
        n_bins (int): Number of bins for large time scale (PSTH)
        n_bins_small (int): Number of bins for small time scale (noise correlation)
        rep_length (float): Length of each repetition in seconds

    Returns:
        tuple: Contains:
            - binned_spikes (np.ndarray): Shape (nb_repetitions, n_bins)
            - spike_counts_small_bin (np.ndarray): Shape (nb_repetitions, n_bins_small)
    """
    binned_spikes = np.empty((nb_repetitions, n_bins))
    spike_counts_small_bin = np.empty((nb_repetitions, n_bins_small))
    
    for i in range(nb_repetitions):
        binned_spikes[i, :] = np.histogram(
            spike_trains[i], 
            bins=n_bins, 
            range=(0, rep_length)
        )[0]
        
        spike_counts_small_bin[i, :] = np.histogram(
            spike_trains[i], 
            bins=n_bins_small, 
            range=(0, rep_length)
        )[0]
    
    return binned_spikes, spike_counts_small_bin


def compute_psth_and_noise(binned_spikes: np.ndarray, spike_counts_small_bin: np.ndarray, 
                           nb_repetitions: int, time_bin: float, rep_length: float):
    """Compute PSTH and noise statistics from binned spike data.

    Args:
        binned_spikes (np.ndarray): Binned spikes at large time scale
        spike_counts_small_bin (np.ndarray): Binned spikes at small time scale
        nb_repetitions (int): Number of stimulus repetitions
        time_bin (float): Duration of time bin in seconds
        rep_length (float): Length of repetition in seconds

    Returns:
        tuple: Contains:
            - psth (np.ndarray): Peri-stimulus time histogram (firing rate in Hz)
            - mean_spikes_count (np.ndarray): Mean spike count across repetitions (small bins)
            - noise_small_bin (np.ndarray): Deviation from mean at small time scale
            - noise_large_bin (np.ndarray): Deviation from mean at large time scale
    """
    # Compute PSTH (sum across repetitions, convert to firing rate)
    psth = np.sum(binned_spikes, axis=0) / time_bin
    
    # Compute mean spike count at small time scale
    mean_spikes_count = np.sum(spike_counts_small_bin, axis=0) / nb_repetitions
    
    # Compute noise (deviation from mean)
    noise_small_bin = np.subtract(spike_counts_small_bin, mean_spikes_count)
    noise_large_bin = np.subtract(binned_spikes, psth / rep_length)
    
    return psth, mean_spikes_count, noise_small_bin, noise_large_bin


def compute_chirp_rasters(cells: list, spike_times: list, stim_onsets: np.ndarray, 
                          old: bool = False, n_bins: int = 800, n_bins_small: int = 16000):
    """Compute chirp stimulus responses including rasters, PSTH, and noise correlations.

    Args:
        cells (list): List of cell/cluster IDs
        spike_times (list[np.ndarray]): Spike times for each cell
        stim_onsets (np.ndarray): Stimulus onset times in seconds
        old (bool): If True, use old chirp parameters. Default False.
        n_bins (int): Number of bins for PSTH. Default 800.
        n_bins_small (int): Number of bins for noise analysis. Default 16000.

    Returns:
        dict: Nested dictionary with structure:
            cell_data[cell_id] = {
                'spike_times': original spike times,
                'repeated_sequences_times': list of [start, end] for each rep,
                'spike_trains': aligned spike trains for each rep,
                'psth': peri-stimulus time histogram (Hz),
                'mean_spikes_count_small_bin': mean spike count at fine timescale,
                'spikes_counts_small_bin': spike counts for each rep at fine timescale,
                'noise_small_bin': deviation from mean at fine timescale,
                'noise_large_bin': deviation from mean at coarse timescale
            }
    """
    print("Extracting cells responses to Chirp stimulus\n")
    
    # Get stimulus parameters
    nb_repetitions, n_bins, rep_length, time_bin = get_chirp_parameters(old)
    
    cell_data = {}

    for idx, cell_nb in tqdm(enumerate(cells[:]), desc="Extraction"):
        if cell_nb not in cell_data:
            cell_data[cell_nb] = {}
        
        euler_sptimes = spike_times[idx]
        
        # Extract repeated sequence times
        repeated_sequences_times = extract_repeated_sequence_times(
            stim_onsets, nb_repetitions, old
        )
        
        # Align spike trains to stimulus onset
        spike_trains = align_spike_trains(euler_sptimes, repeated_sequences_times)
        
        # Bin spikes at multiple time scales
        binned_spikes, spike_counts_small_bin = compute_binned_responses(
            spike_trains, nb_repetitions, n_bins, n_bins_small, rep_length
        )
        
        # Compute PSTH and noise statistics
        psth, mean_spikes_count, noise_small_bin, noise_large_bin = compute_psth_and_noise(
            binned_spikes, spike_counts_small_bin, nb_repetitions, time_bin, rep_length
        )
        
        # Store results
        cell_data[cell_nb]["spike_times"] = spike_times
        cell_data[cell_nb]["repeated_sequences_times"] = repeated_sequences_times
        cell_data[cell_nb]["spike_trains"] = spike_trains
        cell_data[cell_nb]["psth"] = psth
        cell_data[cell_nb]["mean_spikes_count_small_bin"] = mean_spikes_count
        cell_data[cell_nb]["spikes_counts_small_bin"] = spike_counts_small_bin
        cell_data[cell_nb]["noise_small_bin"] = noise_small_bin
        cell_data[cell_nb]["noise_large_bin"] = noise_large_bin

    return cell_data


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def load_chirp_stimulus_vector(old: bool):
    """Load the chirp stimulus vector file.

    Args:
        old (bool): If True, load old chirp vector. If False, load new chirp vector.

    Returns:
        tuple: Contains:
            - euler_vec (np.ndarray): Chirp stimulus vector
            - rep_length (float): Length of repetition in seconds
            - n_bins (int): Number of bins
            - n_reps (int): Number of repetitions
    """
    if old:
        vec_path = os.path.join('./ressources', "EulerStim180530.vec")
        euler_vec = -np.genfromtxt(vec_path)
        rep_length = 25
        n_bins = 625
        n_reps = 30
    else:
        vec_path = os.path.join('./ressources', "Euler_50Hz_20reps_1024x768pix.vec")
        euler_vec = np.genfromtxt(vec_path)
        rep_length = 32
        n_bins = 800
        n_reps = 20
    
    return euler_vec, rep_length, n_bins, n_reps


def create_chirp_raster_figure(cell_nb: int, cell_data: dict, sta_results: dict, 
                               euler_vec: np.ndarray, rep_length: float, 
                               n_bins: int, old: bool):
    """Create a single figure showing chirp raster, PSTH, and STA for one cell.

    Args:
        cell_nb (int): Cell/cluster ID
        cell_data (dict): Dictionary containing cell's chirp response data
        sta_results (dict): Dictionary containing STA analysis results
        euler_vec (np.ndarray): Chirp stimulus vector
        rep_length (float): Length of repetition in seconds
        n_bins (int): Number of time bins
        old (bool): If True, use old chirp indexing

    Returns:
        matplotlib.figure.Figure: Figure object with chirp visualization
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(19, 6))
    gs = GridSpec(8, 19, left=0.1, right=0.9, bottom=0.1, top=0.9,
                  wspace=0.4, hspace=0, figure=fig)
    
    time_bin = rep_length / n_bins
    
    # Plot chirp stimulus
    ax = fig.add_subplot(gs[0:2, :-3])
    if old:
        ax.plot(np.linspace(0, rep_length, 999), euler_vec[1:1000, 1], color='k')
    else:
        ax.plot(np.linspace(0, rep_length, 1600), euler_vec[151:1751, 1], color='k')
    ax.set_ylabel('Chirp Stimulus')
    ax.set_yticks([])
    ax.set_title(f'Cluster {cell_nb}')
    ax.set_xlim([0, rep_length])
    
    # Plot chirp raster
    ax = fig.add_subplot(gs[2:5, :-3])
    ax.eventplot(cell_data[cell_nb]["spike_trains"], color='k', lw=1, linelengths=1)
    ax.set_ylabel('#Trial')
    ax.set_xlim([0, rep_length])
    
    # Plot chirp PSTH
    ax = fig.add_subplot(gs[5:, :-3])
    ax.step(np.linspace(0, rep_length, n_bins), cell_data[cell_nb]["psth"])
    ax.set_ylabel(f'Firing rate Hz \n ({int(time_bin*1000)} ms time bin)')
    ax.set_xlabel("Time (s)")
    ax.set_xlim([0, rep_length])
    
    # Plot spatial STA
    ax = fig.add_subplot(gs[0:3, -3:])
    ax.set_title("STA", fontsize=12)
    spatial = sta_results[cell_nb]['center_analyse']['Spatial']
    spatial = spatial**2 * np.sign(spatial)
    cmap = 'RdBu_r'
    image = ax.imshow(spatial, cmap=cmap, interpolation='gaussian')
    abs_max = 0.5 * max(np.max(spatial), abs(np.min(spatial)))
    image.set_clim(-abs_max, abs_max)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig


def plot_chirp_rasters(cells: list, cell_data: dict, CT_directory: str, 
                       check_directory: str, old: bool = False):
    """Generate and save chirp raster plots for all cells.

    Args:
        cells (list): List of cell/cluster IDs to plot
        cell_data (dict): Dictionary containing chirp response data for all cells
        CT_directory (str): Path to cell typing output directory
        check_directory (str): Path to checkerboard analysis directory containing STA results
        old (bool): If True, use old chirp parameters. Default False.

    Returns:
        None. Saves PNG files to CT_directory/Chirp_rasters+STA/
    """
    # Create output directory
    fig_directory = os.path.normpath(os.path.join(CT_directory, 'Chirp_rasters+STA'))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)
    
    # Load STA results
    sta_results = np.load(
        os.path.join(check_directory, 'sta_data_3D_fitted.pkl'),
        allow_pickle=True
    )
    
    # Load chirp stimulus vector
    euler_vec, rep_length, n_bins, n_reps = load_chirp_stimulus_vector(old)
    
    print(f"Saving Chirp raster plots in : {fig_directory} \n")
    
    # Generate plots for each cell
    for cell_nb in tqdm(cells[:]):
        fig = create_chirp_raster_figure(
            cell_nb, cell_data, sta_results, euler_vec, rep_length, n_bins, old
        )
        
        fsave = os.path.join(fig_directory, f'{cell_nb}_Chirp_raster+STA')
        fig.savefig(fsave + '.png', format='png', dpi=90)
        plt.close(fig)
    
    print('--- Cell Done ---')


# =============================================================================
# CELL SELECTION FUNCTIONS
# =============================================================================

def load_or_initialize_cell_selection(CT_directory: str, exp: str):
    """Load previous cell selection if exists, otherwise initialize empty lists.

    Args:
        CT_directory (str): Path to cell typing output directory
        exp (str): Experiment name

    Returns:
        tuple: Contains:
            - selected_cells (list): All selected cells
            - selected_cells_sta (list): Cells selected based on STA quality
            - selected_cells_chirp (list): Cells selected based on chirp response quality
    """
    all_selected_cells_file = os.path.normpath(os.path.join(
        CT_directory, 
        f'{exp}_selected_cells_for_clustering.pkl'
    ))
    
    if os.path.isfile(all_selected_cells_file):
        print(f"Loading previous selection from  : {all_selected_cells_file}")
        all_selected_cells = utils.load_obj(all_selected_cells_file)
        selected_cells = all_selected_cells["selected_cells"]
        selected_cells_sta = all_selected_cells["selected_cells_sta"]
        selected_cells_chirp = all_selected_cells["selected_cells_chirp"]
    else:
        selected_cells = []
        selected_cells_sta = []
        selected_cells_chirp = []
    
    return selected_cells, selected_cells_sta, selected_cells_chirp


def select_and_save_cells_for_clustering(good_sta_cells: list, good_chirp_cells: list, 
                                        CT_directory: str, params: dict):
    """Update cell selection for clustering analysis.

    Args:
        good_sta_cells (list): Cell IDs with good STA quality (or empty list)
        good_chirp_cells (list): Cell IDs with good chirp responses (or empty list)
        CT_directory (str): Path to cell typing output directory
        params (dict): Experiment parameters containing 'exp' field

    Returns:
        tuple: Contains:
            - selected_cells (list): Final list of all selected cells
            - selected_cells_sta (list): Cells selected based on STA quality
            - selected_cells_chirp (list): Cells selected based on chirp quality

    Note:
        Calls cell_selection_for_clustering() which should be defined in utils.
        This function provides an interactive interface for manual cell selection.
    """
    exp = params.exp
    
    # Load or initialize selection
    selected_cells, selected_cells_sta, selected_cells_chirp = load_or_initialize_cell_selection(
        CT_directory, exp
    )
    
    # Update with new selections
    if good_sta_cells:
        selected_cells_sta = good_sta_cells
    if good_chirp_cells:
        selected_cells_chirp = good_chirp_cells
    
    # Interactive cell selection
    fig_directory = os.path.normpath(os.path.join(CT_directory, 'Chirp_rasters+STA'))
    selected_cells, selected_cells_sta, selected_cells_chirp = cell_selection_for_clustering(
        cells, 
        CT_directory_path=fig_directory,
        selected_cells_sta=selected_cells_sta,
        selected_cells_chirp=selected_cells_chirp
    )
    
    print(f"Selected {len(selected_cells)} cells.")
    
    return selected_cells, selected_cells_sta, selected_cells_chirp


def modify_cells_for_clustering(selected_cells_sta: list, selected_cells_sta_to_add: list,
                                selected_cells_sta_to_remove: list, selected_cells_chirp: list,
                                selected_cells_chirp_to_add: list, selected_cells_chirp_to_remove: list,
                                remove_any_way: list, CT_directory: str, params: dict):
    """Modify selected cells by adding/removing specific cells and save updated selection.

    Args:
        selected_cells_sta (list): Current STA-selected cells
        selected_cells_sta_to_add (list): Cell IDs to add to STA selection
        selected_cells_sta_to_remove (list): Cell IDs to remove from STA selection
        selected_cells_chirp (list): Current chirp-selected cells
        selected_cells_chirp_to_add (list): Cell IDs to add to chirp selection
        selected_cells_chirp_to_remove (list): Cell IDs to remove from chirp selection
        remove_any_way (list): Cell IDs to remove from all selections
        CT_directory (str): Path to cell typing output directory
        params (dict): Experiment parameters containing 'exp' field

    Returns:
        tuple: Contains:
            - selected_cells (list): Final combined selection
            - selected_cells_sta (list): Updated STA-selected cells
            - selected_cells_chirp (list): Updated chirp-selected cells

    Note:
        Saves updated selection to '{exp}_selected_cells_for_clustering.pkl'
    """
    exp = params.exp
    
    # Update chirp selection
    selected_cells_chirp = list(set([
        idx for idx in selected_cells_chirp + selected_cells_chirp_to_add 
        if idx not in selected_cells_chirp_to_remove + remove_any_way
    ]))
    
    # Update STA selection
    selected_cells_sta = list(set([
        idx for idx in selected_cells_sta + selected_cells_sta_to_add 
        if idx not in selected_cells_sta_to_remove + remove_any_way
    ]))
    
    # Re-run cell selection interface
    selected_cells, selected_cells_sta, selected_cells_chirp = cell_selection_for_clustering(
        cells,
        CT_directory_path=CT_directory,
        selected_cells_sta=list(set(selected_cells_sta)),
        selected_cells_chirp=list(set(selected_cells_chirp))
    )
    
    # Save updated selection
    fsave = os.path.join(CT_directory, f'{exp}_selected_cells_for_clustering')
    save_obj({
        "selected_cells": selected_cells,
        "selected_cells_sta": selected_cells_sta,
        "selected_cells_chirp": selected_cells_chirp
    }, fsave)
    
    return selected_cells, selected_cells_sta, selected_cells_chirp


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def bin_spikes_for_pca(cell_data: dict, selected_cells: list, n_rep: int = 20,
                       nt: float = 32, dt: float = 0.04):
    """Bin spike trains for PCA analysis.

    Args:
        cell_data (dict): Dictionary containing spike train data for all cells
        selected_cells (list): List of cell IDs to include in analysis
        n_rep (int): Number of stimulus repetitions. Default 20.
        nt (float): Total length of stimulus in seconds. Default 32.
        dt (float): Bin size in seconds. Default 0.04.

    Returns:
        np.ndarray: Binned spikes with shape (n_cells, n_time_bins, n_repetitions)
    """
    n_cells = len(selected_cells)
    time_bins = np.arange(0, nt + dt, dt)
    spikes = np.zeros((n_cells, int(nt/dt), n_rep))
    
    for cell_index in range(n_cells):
        cell_id = selected_cells[cell_index]
        spike_cell = cell_data[cell_id]["spike_trains"]
        
        for rep in range(n_rep):
            temp = np.histogram(spike_cell[rep], bins=time_bins)
            spikes[cell_index, :, rep] = temp[0]
    
    return spikes


def compute_psth_pca(spikes: np.ndarray, n_components: int, sparse: bool = False):
    """Compute PCA on PSTH (mean spike response).

    Args:
        spikes (np.ndarray): Binned spikes, shape (n_cells, n_time_bins, n_repetitions)
        n_components (int): Number of PCA components to compute
        sparse (bool): If True, use SparsePCA. If False, use standard PCA. Default False.

    Returns:
        tuple: Contains:
            - psth_pca (np.ndarray): PCA-transformed PSTH, shape (n_cells, n_components)
            - pca_transformer: Fitted PCA transformer object
            - psth_z (np.ndarray): Z-scored PSTH, shape (n_cells, n_time_bins)
    """
    import scipy as sc
    from sklearn.decomposition import PCA, SparsePCA
    
    # Compute PSTH and z-score
    psth = np.mean(spikes, 2)
    psth_z = sc.stats.zscore(psth, 1)
    
    # Fit PCA
    if sparse:
        pca_transformer = SparsePCA(n_components, random_state=0).fit(psth_z)
    else:
        pca_transformer = PCA(n_components).fit(psth_z)
    
    psth_pca = pca_transformer.transform(psth_z)
    
    return psth_pca, pca_transformer, psth_z


def extract_sta_time_courses(sta_results: dict, selected_cells: list):
    """Extract temporal components of STAs for selected cells.

    Args:
        sta_results (dict): Dictionary containing STA analysis results
        selected_cells (list): List of cell IDs to extract STAs for

    Returns:
        np.ndarray: STA time courses, shape (n_cells, 21)
    """
    n_cells = len(selected_cells)
    STA_time_course = np.zeros((n_cells, 21))
    
    for cell_index in range(n_cells):
        cell_id = selected_cells[cell_index]
        TempSTA_cell = sta_results[cell_id]['center_analyse']['Temporal'][-21:]
        STA_time_course[cell_index] = TempSTA_cell
    
    return STA_time_course


def compute_sta_pca(STA_time_course: np.ndarray, n_components: int):
    """Compute PCA on STA temporal components.

    Args:
        STA_time_course (np.ndarray): STA time courses, shape (n_cells, n_timepoints)
        n_components (int): Number of PCA components to compute (use 0 to skip STA PCA)

    Returns:
        tuple: Contains:
            - sta_tc_pca (np.ndarray or None): PCA-transformed STA, shape (n_cells, n_components)
                                               or None if n_components == 0
            - pca_transformer2 (PCA or None): Fitted PCA transformer or None if n_components == 0
    """
    import scipy as sc
    from sklearn.decomposition import PCA
    
    # Z-score STA time courses
    sta_tc = sc.stats.zscore(STA_time_course[:, :], 1)
    
    if n_components > 0:
        pca_transformer2 = PCA(n_components).fit(sta_tc)
        sta_tc_pca = pca_transformer2.transform(sta_tc)
        return sta_tc_pca, pca_transformer2
    else:
        return None, None


def compute_ellipse_sizes(sta_results: dict, selected_cells: list):
    """Extract and normalize RF ellipse sizes for selected cells.

    Args:
        sta_results (dict): Dictionary containing STA analysis results with ellipse parameters
        selected_cells (list): List of cell IDs to extract ellipse sizes for

    Returns:
        np.ndarray: Normalized ellipse sizes, shape (n_cells,), values in range [0, 1]
    """
    n_cells = len(selected_cells)
    ell_size = np.zeros(n_cells)
    
    for cell_index in range(n_cells):
        cell_id = selected_cells[cell_index]
        width = sta_results[cell_id]['center_analyse']['EllipseCoor'][3]
        height = sta_results[cell_id]['center_analyse']['EllipseCoor'][4]
        ell_size[cell_index] = np.abs(np.pi * width * height)
    
    # Normalize to [0, 1] range
    ell_size_normalized = -np.ones(n_cells)
    temp = ell_size[:] - ell_size[:].min()
    ell_size_normalized[:] = temp / temp.max()
    
    return ell_size_normalized


def build_clustering_dataset(psth_pca: np.ndarray, sta_tc_pca: np.ndarray, 
                             ell_size_normalized: np.ndarray, n_components_psth: int,
                             n_components_sta_tc: int):
    """Combine PCA features and ellipse sizes into clustering dataset.

    Args:
        psth_pca (np.ndarray): PCA components from PSTH, shape (n_cells, n_components_psth)
        sta_tc_pca (np.ndarray or None): PCA components from STA, shape (n_cells, n_components_sta)
                                         or None if not using STA
        ell_size_normalized (np.ndarray): Normalized ellipse sizes, shape (n_cells,)
        n_components_psth (int): Number of PSTH PCA components
        n_components_sta_tc (int): Number of STA PCA components (0 if not using STA)

    Returns:
        np.ndarray: Combined dataset for clustering, 
                   shape (n_cells, n_components_psth + n_components_sta_tc + 1)
    """
    n_cells = psth_pca.shape[0]
    cluster_dataset = np.zeros((n_cells, n_components_psth + n_components_sta_tc + 1))
    
    # Add PSTH PCA components
    cluster_dataset[:, :n_components_psth] = psth_pca
    
    # Add STA PCA components if available
    if n_components_sta_tc > 0 and sta_tc_pca is not None:
        cluster_dataset[:, n_components_psth:n_components_psth + n_components_sta_tc] = sta_tc_pca
    
    # Add ellipse size as last feature
    cluster_dataset[:, -1] = ell_size_normalized
    
    return cluster_dataset


def perform_agglomerative_clustering(cluster_dataset: np.ndarray, dist_thres: float):
    """Perform hierarchical agglomerative clustering.

    Args:
        cluster_dataset (np.ndarray): Feature matrix for clustering, shape (n_cells, n_features)
        dist_thres (float): Distance threshold for clustering. Adjust to get ~50 clusters.

    Returns:
        AgglomerativeClustering: Fitted clustering model with labels in model.labels_
    """
    from sklearn.cluster import AgglomerativeClustering
    
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=None)
    model = model.fit(cluster_dataset)
    
    return model


def plot_pca_variance(pca_transformer, n_components: int, title: str):
    """Plot cumulative explained variance for PCA.

    Args:
        pca_transformer: Fitted PCA transformer object with explained_variance_ratio_ attribute
        n_components (int): Number of components to plot
        title (str): Title for the plot (e.g., 'Chirp PSTH' or 'STA')

    Returns:
        None. Displays matplotlib plot.
    """
    plt.plot(np.arange(n_components) + 1, 
             np.cumsum(pca_transformer.explained_variance_ratio_) * 100)
    plt.axhline(y=80, color='k')
    plt.xlabel(f'Number of PCs from {title}')
    plt.ylabel('% of Cumulative Explained Variance')
    plt.show()


def plot_dendrogram(model, truncate_mode: str = 'level', p: int = 0):
    """Create a dendrogram plot for hierarchical clustering.

    Args:
        model: Fitted AgglomerativeClustering model
        truncate_mode (str): Mode for truncating dendrogram. Default 'level'.
        p (int): Depth parameter for truncation. Default 0.

    Returns:
        None. Displays matplotlib plot.
    
    Note:
        This is a helper function used by plot_clustering_diagnostics.
    """
    from scipy.cluster.hierarchy import dendrogram
    
    # Create linkage matrix from sklearn model
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ]).astype(float)
    
    dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p)


def plot_cluster_centroids(psth_z: np.ndarray, cluster_labels: np.ndarray):
    """Plot average PSTH for each cluster.

    Args:
        psth_z (np.ndarray): Z-scored PSTHs, shape (n_cells, n_time_bins)
        cluster_labels (np.ndarray): Cluster assignment for each cell

    Returns:
        None. Displays matplotlib plot.
    """
    n_clusts = len(np.unique(cluster_labels))
    plt.figure()
    
    for iclust in range(n_clusts):
        idx_cluster = np.where(cluster_labels == iclust)[0]
        plt.plot(np.mean(psth_z[idx_cluster, :], 0) + iclust * 5)
    
    plt.xlabel('Time Bin')
    plt.ylabel('Cluster (offset for visibility)')
    plt.title('Cluster Centroids')
    plt.show()


def plot_clustering_diagnostics(model, psth_z: np.ndarray, dist_thres: float,
                                pca_transformer=None, pca_transformer2=None,
                                n_components_psth: int = 0, n_components_sta_tc: int = 0,
                                sparse: bool = False):
    """Plot diagnostic figures for clustering analysis.

    Args:
        model: Fitted AgglomerativeClustering model
        psth_z (np.ndarray): Z-scored PSTHs, shape (n_cells, n_time_bins)
        dist_thres (float): Distance threshold used for clustering
        pca_transformer: Fitted PCA transformer for PSTH (or None if sparse)
        pca_transformer2: Fitted PCA transformer for STA (or None)
        n_components_psth (int): Number of PSTH PCA components. Default 0.
        n_components_sta_tc (int): Number of STA PCA components. Default 0.
        sparse (bool): Whether SparsePCA was used. Default False.

    Returns:
        None. Displays matplotlib plots.
    """
    # Plot PCA variance if not using sparse PCA
    if not sparse and pca_transformer is not None:
        plot_pca_variance(pca_transformer, n_components_psth, 'Chirp PSTH')
        
        if n_components_sta_tc > 0 and pca_transformer2 is not None:
            plot_pca_variance(pca_transformer2, n_components_sta_tc, 'STA')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, truncate_mode='level', p=0)
    plt.axhline(dist_thres, color='k', linestyle='--', label=f'Threshold: {dist_thres}')
    plt.xlabel("Number of points in node (or index of point if no parenthesis)")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()
    
    # Plot cluster centroids
    plot_cluster_centroids(psth_z, model.labels_)
    
    print(f'Number of clusters: {len(np.unique(model.labels_))}')


def run_cell_typing_AC(dist_thres: float, n_components_psth: int, n_components_sta_tc: int,
                       cell_data: dict, selected_cells: list, check_directory: str,
                       sparse: bool = False):
    """Run cell typing using Agglomerative Clustering on PSTH and STA features.

    Args:
        dist_thres (float): Distance threshold for clustering. Adjust to get ~50 clusters.
        n_components_psth (int): Number of PCA components for chirp PSTH. 
                                Should explain ~80% variance.
        n_components_sta_tc (int): Number of PCA components for STA temporal component.
                                  Usually 2 for reliable checkerboard, 1 otherwise, 0 to skip.
        cell_data (dict): Dictionary containing spike train data for all cells
        selected_cells (list): List of cell IDs to include in clustering
        check_directory (str): Path to checkerboard analysis directory with STA results
        sparse (bool): If True, use SparsePCA for PSTH. Default False.

    Returns:
        tuple: Contains:
            - psth_z (np.ndarray): Z-scored PSTHs, shape (n_cells, n_time_bins)
            - sta_results (dict): Loaded STA analysis results
            - model: Fitted AgglomerativeClustering model with cluster labels

    Note:
        Displays diagnostic plots: PCA variance, dendrogram, cluster centroids.
        You want ~80% cumulative variance explained by PCA components.
        Adjust dist_thres to get approximately 50 clusters.
    """
    # Load STA results
    sta_results = np.load(
        os.path.join(check_directory, 'sta_data_3D_fitted.pkl'),
        allow_pickle=True
    )
    
    # Bin spikes for PCA
    spikes = bin_spikes_for_pca(cell_data, selected_cells, n_rep=20, nt=32, dt=0.04)
    
    # Compute PSTH PCA
    psth_pca, pca_transformer, psth_z = compute_psth_pca(spikes, n_components_psth, sparse)
    
    # Extract and compute STA PCA
    STA_time_course = extract_sta_time_courses(sta_results, selected_cells)
    sta_tc_pca, pca_transformer2 = compute_sta_pca(STA_time_course, n_components_sta_tc)
    
    # Compute ellipse sizes
    ell_size_normalized = compute_ellipse_sizes(sta_results, selected_cells)
    
    # Build clustering dataset
    cluster_dataset = build_clustering_dataset(
        psth_pca, sta_tc_pca, ell_size_normalized, 
        n_components_psth, n_components_sta_tc
    )
    
    # Perform clustering
    model = perform_agglomerative_clustering(cluster_dataset, dist_thres)
    
    # Plot diagnostics
    plot_clustering_diagnostics(
        model, psth_z, dist_thres, 
        pca_transformer, pca_transformer2,
        n_components_psth, n_components_sta_tc, sparse
    )
    
    return psth_z, sta_results, model


def save_cluster_number_for_cells(selected_cells: list, cell_data: dict, model):
    """Save each cell's cluster assignment in the cell_data dictionary.

    Args:
        selected_cells (list): List of cell IDs that were included in clustering
        cell_data (dict): Dictionary containing data for all cells
        model: Fitted clustering model with labels_ attribute

    Returns:
        dict: Updated cell_data with 'type' field added for each cell.
              Clustered cells have integer cluster IDs, others have 'Not assigned'.

    Note:
        Run this after performing agglomerative clustering to save results.
        Modifies cell_data in place and also returns it.
    """
    # Assign cluster labels to selected cells
    for cell_index in range(len(selected_cells)):
        cell_nb = selected_cells[cell_index]
        cell_data[cell_nb]["type"] = model.labels_[cell_index]
    
    # Mark non-selected cells as not assigned
    for cell in cell_data.keys():
        if cell not in selected_cells:
            cell_data[cell]["type"] = 'Not assigned'
    
    return cell_data

# =============================================================================
# NOISE CORRELATION FUNCTIONS
# =============================================================================

def correlate_PersonPM(signal1: np.ndarray, signal2: np.ndarray, max_shift: int):
    """Compute Pearson correlation between two signals with time shifts.

    Args:
        signal1 (np.ndarray): First signal
        signal2 (np.ndarray): Second signal
        max_shift (int): Maximum time shift in bins (positive and negative)

    Returns:
        np.ndarray: Correlation values at each time shift, length 2*max_shift + 1

    Note:
        This should be defined in utils or imported from scipy.signal.correlate.
        Placeholder for the correlation function used in the original code.
    """
    # This is a placeholder - the actual implementation should be in utils
    from scipy.signal import correlate
    from scipy.stats import pearsonr
    
    correlations = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            corr = np.corrcoef(signal1[:shift], signal2[-shift:])[0, 1]
        elif shift > 0:
            corr = np.corrcoef(signal1[shift:], signal2[:-shift])[0, 1]
        else:
            corr = np.corrcoef(signal1, signal2)[0, 1]
        correlations.append(corr)
    
    return np.array(correlations)


def compute_pairwise_correlations(cell: int, cell_data: dict, sta_results: dict,
                                  idx_cluster: list, selected_cells: list,
                                  nb_repetitions: int, max_shift: int):
    """Compute noise correlations between one cell and all others in its cluster.

    Args:
        cell (int): Target cell ID
        cell_data (dict): Dictionary containing noise data for all cells
        sta_results (dict): Dictionary containing STA spatial information
        idx_cluster (list): Indices of cells in the same cluster
        selected_cells (list): List of all selected cell IDs
        nb_repetitions (int): Number of stimulus repetitions
        max_shift (int): Maximum time shift for cross-correlation

    Returns:
        tuple: Contains:
            - corrs (list): Time-shifted correlations for each cell pair
            - dist (list): Spatial distances between cell pairs
            - max_corr (list): Maximum correlation (at zero lag) for each pair
    """
    cell1 = np.sum(cell_data[cell]["noise_small_bin"], axis=0) / nb_repetitions
    corrs = []
    dist = []
    max_corr = []
    
    # Get current cell's RF center
    cell_center = np.asarray(sta_results[cell]["center_analyse"]['EllipseCoor'][1:3])
    
    for index_corr in [idx for idx in idx_cluster if selected_cells[idx] != cell]:
        cell_corr = selected_cells[index_corr]
        
        # Compute time-shifted correlation
        cell2 = np.sum(cell_data[cell_corr]["noise_small_bin"], axis=0) / nb_repetitions
        corrs.append(correlate_PersonPM(cell2, cell1, max_shift=max_shift))
        
        # Compute spatial distance
        cell_corr_center = np.asarray(
            sta_results[cell_corr]["center_analyse"]['EllipseCoor'][1:3]
        )
        dist.append(np.linalg.norm(cell_corr_center - cell_center))
        
        # Compute zero-lag correlation on large bins
        corr_zero_lag = np.corrcoef(
            np.sum(cell_data[cell]["noise_large_bin"], axis=0) / nb_repetitions,
            np.sum(cell_data[cell_corr]["noise_large_bin"], axis=0) / nb_repetitions
        )[0, 1]
        max_corr.append(corr_zero_lag)
    
    return corrs, dist, max_corr


def compute_intracluster_crosscorr(cell_data: dict, sta_results: dict, selected_cells: list, n_bins: int, max_shift: int, n_repetitions = 30) -> dict:
    """Compute noise correlations within each cluster.

    Args:
        cell_data (dict): Dictionary containing spike data for all cells
        sta_results (dict): Dictionary containing STA spatial information
        selected_cells (list): List of cell IDs included in clustering
        n_bins (int): Number of time bins (not used, kept for compatibility)
        max_shift (int): Maximum time shift for cross-correlation analysis
        n_repetitions (int): Number of stimulus repetitions

    Returns:
        dict: Updated cell_data with added fields for each cell:
            - 'corrs': List of time-shifted correlations with cluster members
            - 'mean_corr': Mean correlation across cluster members
            - 'distances': Spatial distances to cluster members
            - 'max_corr': Zero-lag correlations with cluster members

    Note:
        Only computes correlations for cells that were assigned to clusters.
        Cells marked as 'Not assigned' are skipped.
    """
    # Get stimulus parameters
    nb_repetitions = n_repetitions
    
    # Get unique cluster IDs (excluding 'Not assigned')
    cluster_ids = list(set([
        cell_data[cell]["type"] 
        for cell in cell_data.keys() 
        if cell_data[cell]["type"] != 'Not assigned'
    ]))
    
    # Compute correlations within each cluster
    for icluster in tqdm(cluster_ids, desc="Computing noise correlations within cell types"):
        # Get indices of cells in this cluster
        idx_cluster = sorted(list(np.where(np.asarray([
            cell_data[cell]["type"] 
            for cell in selected_cells 
            if cell_data[cell]["type"] != 'Not assigned'
        ]) == icluster)[0]))
        
        # Compute pairwise correlations for each cell in cluster
        for index in idx_cluster:
            cell = selected_cells[index]
            
            corrs, dist, max_corr = compute_pairwise_correlations(
                cell, cell_data, sta_results, idx_cluster, 
                selected_cells, nb_repetitions, max_shift
            )
            
            # Store results
            cell_data[cell]["corrs"] = corrs
            cell_data[cell]["mean_corr"] = np.mean(np.asarray(corrs), axis=0) if corrs else []
            cell_data[cell]["distances"] = dist
            cell_data[cell]["max_corr"] = max_corr
    
    return cell_data

# =============================================================================
# SUMMARY FIGURE FUNCTIONS
# =============================================================================

def load_dg_and_stimulus_data(exp: str, old: bool):
    """Load direction/orientation selectivity data and chirp stimulus vector.

    Args:
        exp (str): Experiment name
        old (bool): If True, load old chirp. If False, load new chirp.

    Returns:
        tuple: Contains:
            - DG_set (dict): Direction/orientation selectivity data
            - euler_vec (np.ndarray): Chirp stimulus vector
    """
    # Load direction selectivity data
    DG_set = utils.load_obj(os.path.join(
        utils.find_Analysis_Directory(dir_type="DG"),
        f'DG_data_exp{exp}'
    ))
    
    # Load chirp stimulus vector
    if old:
        vec_path = os.path.join('./ressources', "EulerStim180530.vec")
        euler_vec = -np.genfromtxt(vec_path)
    else:
        vec_path = os.path.join('./ressources', "Euler_50Hz_20reps_1024x768pix.vec")
        euler_vec = np.genfromtxt(vec_path)
    
    return DG_set, euler_vec


def plot_sta_with_ellipse(ax, spatial_sta: np.ndarray, ellipse_params: list):
    """Plot spatial STA with ellipse overlay.

    Args:
        ax: Matplotlib axis object
        spatial_sta (np.ndarray): 2D spatial STA
        ellipse_params (list): Ellipse parameters [amplitude, x0, y0, sigma_x, sigma_y, theta]

    Returns:
        matplotlib.axes.Axes: Modified axis object
    
    Note:
        Assumes gaussian2D function is available (should be in utils).
    """
    x0 = ellipse_params[1]
    y0 = ellipse_params[2]
    
    # Apply nonlinear transformation for visualization
    spatial = spatial_sta**2 * np.sign(spatial_sta)
    
    cmap = 'RdBu_r'
    image = ax.imshow(spatial, cmap=cmap, interpolation='gaussian')
    
    # Set color limits
    abs_max = 0.5 * max(np.max(spatial), abs(np.min(spatial)))
    image.set_clim(-abs_max, abs_max)
    
    ax.set_xlim(x0 - 4, x0 + 4)
    ax.set_ylim(y0 + 4, y0 - 4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax


def plot_orientation_tuning(ax, DG_data: dict):
    """Plot orientation/direction tuning on polar axis.

    Args:
        ax: Matplotlib polar axis object
        DG_data (dict): Direction selectivity data containing:
            - 'Tuning': Array of responses at each orientation
            - 'atune': Preferred angle
            - 'Rtune': Response magnitude at preferred angle

    Returns:
        None. Modifies axis in place.
    """
    theta = np.linspace(0, 2 * np.pi, 9)
    
    # Set up polar grid
    lines, labels = plt.thetagrids(range(0, 360, int(360/8)), np.arange(0, 360, 45))
    
    # Plot tuning curve
    ax.plot(theta, DG_data['Tuning'])
    ax.fill(theta, DG_data['Tuning'], 'b', alpha=0.1)
    
    # Plot preferred direction
    ax.plot([DG_data['atune'], DG_data['atune']], 
            [0, DG_data['Rtune']], 'b-')
    ax.plot([DG_data['atune']], [DG_data['Rtune']], 'bo')
    
    # Format axis
    ax.set_yticks([0, 0.333, 0.666, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([0, '', '', 135, '', 225, '', ''])
    ax.set_ylim([0, 1])


def plot_distance_correlation_summary(ax, cum_dist: list, cum_corr: list):
    """Plot summary of correlation vs. distance for all cell pairs in cluster.

    Args:
        ax: Matplotlib axis object
        cum_dist (list): List of all pairwise distances in cluster
        cum_corr (list): List of all pairwise correlations in cluster

    Returns:
        None. Modifies axis in place.
    """
    if not cum_dist:
        ax.set_visible(False)
        return
    
    sorted_dists, sorted_max_corr = zip(*sorted(zip(cum_dist, cum_corr)))
    
    n_bin = 10
    xlim = np.linspace(min(sorted_dists), max(sorted_dists), n_bin + 1)
    
    # Compute binned statistics
    mean = [
        np.mean(np.asarray(sorted_max_corr)[
            np.where(np.logical_and(sorted_dists >= xlim[i], 
                                   sorted_dists <= xlim[i+1]))[0]
        ]) 
        for i in range(len(xlim) - 1)
    ]
    
    x_pos = np.linspace(min(sorted_dists), max(sorted_dists), n_bin)
    
    # Plot mean trend
    ax.plot(x_pos[np.isfinite(mean)], 
            np.asarray(mean)[np.isfinite(mean)], 
            color='#1f77b4', linewidth=2)
    
    ax.set_title("Correlations")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Correlation")


def create_single_cell_row(fig, gs, line: int, cell_nb: int, cell_data: dict,
                           sta_results: dict, DG_set: dict, selected_cells: list,
                           ax0, ax_dist_corr, cum_dist: list, cum_corr: list):
    """Create one row of the cluster summary figure for a single cell.

    Args:
        fig: Matplotlib figure object
        gs: GridSpec object for subplot layout
        line (int): Row number for this cell
        cell_nb (int): Cell ID
        cell_data (dict): Dictionary with cell's chirp response data
        sta_results (dict): Dictionary with cell's STA data
        DG_set (dict): Direction selectivity data
        selected_cells (list): List of all selected cells
        ax0: Axis for ellipse overlay plot
        ax_dist_corr: Axis for distance-correlation plot
        cum_dist (list): Cumulative distances (modified in place)
        cum_corr (list): Cumulative correlations (modified in place)

    Returns:
        tuple: Contains:
            - STAs (np.ndarray): Temporal STA (to be accumulated)
            - cum_dist (list): Updated cumulative distances
            - cum_corr (list): Updated cumulative correlations
    """
    # Plot temporal STA (individual)
    ax = fig.add_subplot(gs[line, 2])
    ax.axis("off")
    ax.set_aspect(0.175)
    temporal_sta = sta_results[cell_nb]['center_analyse']['Temporal'][-21:]
    ax.step(np.linspace(-21/30, 0, 21), temporal_sta, 'k', lw=3)
    ax.axhline(0, color='k', lw=0.5)
    
    # Plot chirp PSTH
    ax = fig.add_subplot(gs[line, 4:8])
    ax.plot(np.linspace(0, 32, 800), cell_data[cell_nb]["psth"])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    plt.locator_params(axis="y", nbins=3)
    
    # Plot correlations if available
    if cell_data[cell_nb]["corrs"] != []:
        sorted_dists, sorted_max_corr = zip(*sorted(
            zip(cell_data[cell_nb]["distances"], cell_data[cell_nb]["max_corr"])
        ))
        cum_dist += list(sorted_dists)
        cum_corr += list(sorted_max_corr)
        
        # Scatter plot for this cell
        ax_dist_corr.scatter(sorted_dists, sorted_max_corr, marker='o', 
                            linewidths=1, color='lightblue', alpha=0.5)
        ax_dist_corr.plot(sorted_dists, sorted_max_corr, linestyle='-', 
                         linewidth=1, alpha=0.5)
        
        # Plot cross-correlogram
        ax = fig.add_subplot(gs[line, 8:])
        mean_corr = cell_data[cell_nb]["mean_corr"]
        shift_range = np.arange(-int(len(mean_corr)/2), int(len(mean_corr)/2) + 1, 1)
        plt.plot(shift_range, mean_corr, linewidth=1)
        plt.fill_between(shift_range, 
                        np.min(np.asarray(cell_data[cell_nb]["corrs"]), axis=0),
                        np.max(np.asarray(cell_data[cell_nb]["corrs"]), axis=0),
                        alpha=0.35)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Plot spatial STA
    ax = fig.add_subplot(gs[line, 1])
    parameters = sta_results[cell_nb]['center_analyse']['EllipseCoor']
    ax = plot_sta_with_ellipse(ax, sta_results[cell_nb]['center_analyse']['Spatial'], 
                               parameters)
    
    # Add ellipse contour to summary plot
    gaussian = gaussian2D(sta_results[cell_nb]['center_analyse']['Spatial'].shape, 
                         *parameters)
    if parameters[0] != 0:
        ax0.contour(np.abs(gaussian), levels=[0.6 * np.max(np.abs(gaussian))],
                   colors='k', linestyles='solid', alpha=0.8)
    
    # Add cell label
    ax = fig.add_subplot(gs[line, 3])
    ax.axis("off")
    ax.text(0.5, 0.5, f'Cell {cell_nb}', horizontalalignment='center', 
            verticalalignment='center', fontsize=12)
    