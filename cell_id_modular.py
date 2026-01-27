import numpy as np
import pickle
import os
import glob
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

def import_data_to_plot(params: dict):
    """Load all data required for generating ID card plots.

    Args:
        params (dict): Experiment parameters from params.py containing:
            - exp (str): Experiment name
            - output_directory (str): Path to analysis output directory
            - phy_directory (str): Path to Phy output directory

    Returns:
        tuple: Contains:
            - exp (str): Experiment name
            - output_directory (str): Output directory path
            - phy_directory (str): Phy directory path
            - check_directory (str): Checkerboard analysis directory
            - DG_directory (str): Direction selectivity directory
            - CT_directory (str): Cell typing directory
            - euler_vec (np.ndarray): Chirp stimulus vector
            - check_rast (dict): Checkerboard raster data
            - DG_data (dict): Direction selectivity data
            - sta_results (dict): STA analysis results
            - cells (list): List of cell IDs
            - Chirp_data (dict): Chirp response data

    Note:
        This function loads all necessary data for creating comprehensive
        cell ID cards. Consider refactoring to return a single dict/object
        for cleaner code organization.
    """
    # Extract parameters
    exp = params.exp
    output_directory = params.output_directory
    phy_directory = params.phy_directory

    # Find analysis directories
    check_directory = utils.find_Analysis_Directory(dir_type="Checkerboard")
    DG_directory = utils.find_Analysis_Directory(dir_type="DG")
    CT_directory = utils.find_Analysis_Directory(dir_type="CellTyping")

    # Load data
    euler_vec = utils.load_chirp_stimulus_vector()
    check_rast = utils.load_checkerboard_rasters(check_directory)
    DG_data = utils.load_direction_selectivity_data(DG_directory, exp)
    sta_results, cells = utils.load_sta_results(check_directory)
    Chirp_data = utils.load_chirp_data(CT_directory, exp)
    return (exp, output_directory, phy_directory, check_directory, DG_directory,
            CT_directory, euler_vec, check_rast, DG_data, sta_results, cells, Chirp_data)


# =============================================================================
# CELL QUALITY FILTERING
# =============================================================================

def filter_cells_by_rpv(cells: list, cell_rpvs: dict, rpv_threshold: float = 0.5):
    """Filter cells based on refractory period violation percentage.

    Args:
        cells (list): List of all cell IDs
        cell_rpvs (dict): Dictionary containing RPV data for each cell with structure:
            cell_rpvs[cell_id] = {'rpv': float, 'isi': array, 'nb_spikes': int, ...}
        rpv_threshold (float): Maximum acceptable RPV percentage. Default 0.5%.

    Returns:
        np.ndarray: Array of cell IDs that pass the RPV threshold

    Note:
        Refractory period violations (RPVs) indicate potential contamination
        from other units. Lower RPV percentages indicate better unit isolation.
    """
    good_cells = []
    
    for cell_nb in cells:
        if cell_rpvs[cell_nb]['rpv'] < rpv_threshold:
            good_cells.append(cell_nb)
    
    return np.array(good_cells)


def select_and_save_good_cells(cells: list, cell_rpvs: dict, output_directory: str,
                               rpv_threshold: float = 0.5):
    """Select and save cells that meet quality criteria based on RPV.

    Args:
        cells (list): List of all cell IDs
        cell_rpvs (dict): Dictionary containing RPV data for each cell
        output_directory (str): Path to save good cells array
        rpv_threshold (float): Maximum acceptable RPV percentage. Default 0.5%.

    Returns:
        np.ndarray: Array of good cell IDs

    Note:
        Saves the good cells array to 'Good_cells.npy' in output_directory.
        Also prints the list of good cells to console.
    """
    good_cells = filter_cells_by_rpv(cells, cell_rpvs, rpv_threshold)
    
    # Save to file
    np.save(os.path.join(output_directory, 'Good_cells'), good_cells)
    
    # Print results
    print(f"Found {len(good_cells)} good cells out of {len(cells)} total cells")
    print(f"Good cells: {good_cells}")
    
    return good_cells


# =============================================================================
# ID CARD PLOTTING - INDIVIDUAL SUBPLOTS
# =============================================================================

def plot_isi_histogram(ax, cell_rpvs: dict, cell_nb: int, rpv_len: float = 2.0):
    """Plot interspike interval (ISI) histogram with RPV threshold.

    Args:
        ax: Matplotlib axis object
        cell_rpvs (dict): Dictionary containing RPV data for the cell
        cell_nb (int): Cell ID
        rpv_len (float): Refractory period length in ms. Default 2.0 ms.

    Returns:
        None. Modifies axis in place.
    """
    ax.hist(cell_rpvs[cell_nb]['isi'] * 1000, bins=100, range=(0, 50))
    
    rpv = cell_rpvs[cell_nb]['rpv']
    nb_spikes = cell_rpvs[cell_nb]['nb_spikes']
    nb_rpv = cell_rpvs[cell_nb]['nb_rpv_spikes']
    
    ax.axvline(rpv_len, lw=0.5, color='k')
    ax.set_title(f"Interspike Interval histogram\n RPV = {round(rpv, 4)}%. "
                f"{int(nb_rpv)}/{nb_spikes} spikes")
    ax.set_xlabel("Interspike time (ms)")
    ax.set_ylabel("Number of spikes")
    
    # Clean up spines
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='k', lw=0.5)


def plot_spatial_sta(ax, sta_results: dict, cell_nb: int):
    """Plot spatial receptive field from STA.

    Args:
        ax: Matplotlib axis object
        sta_results (dict): STA analysis results
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    ax.set_title("Spatial receptive field")
    
    spatial = sta_results[cell_nb]['center_analyse']['Spatial']
    spatial = spatial**2 * np.sign(spatial)
    
    cmap = 'RdBu_r'
    im = ax.imshow(spatial, cmap=cmap, interpolation='gaussian')
    
    abs_max = 0.5 * max(np.max(spatial), abs(np.min(spatial)))
    im.set_clim(-abs_max, abs_max)


def plot_temporal_sta(ax, sta_results: dict, cell_nb: int):
    """Plot temporal receptive field from STA.

    Args:
        ax: Matplotlib axis object
        sta_results (dict): STA analysis results
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    ax.set_title("Temporal receptive field")
    
    temporal_sta = sta_results[cell_nb]['center_analyse']['Temporal'][-21:]
    ax.step(np.linspace(-1, 0, 21), temporal_sta, color='k', lw=3)
    
    ax.set_xlabel("Time (s)")
    ax.axhline(0, color='k', lw=0.5)
    ax.set_yticks([])
    ax.axis('off')


def plot_checkerboard_raster(ax, check_rast: dict, cell_nb: int):
    """Plot raster for repeated white noise sequences (checkerboard).

    Args:
        ax: Matplotlib axis object
        check_rast (dict): Checkerboard raster data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    ax.eventplot(check_rast[cell_nb]["spike_trains"], color='k', alpha=1, linelengths=1)
    ax.set_title("Repeated white noise sequences")
    ax.set_xlabel("Time (s)")
    
    seq_length = (check_rast[cell_nb]["repeated_sequences_times"][0][1] -
                 check_rast[cell_nb]["repeated_sequences_times"][0][0])
    
    ax.set_xlim([0, seq_length])
    ax.set_ylim([0, None])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_checkerboard_psth(ax, check_rast: dict, cell_nb: int):
    """Plot PSTH for repeated white noise sequences.

    Args:
        ax: Matplotlib axis object
        check_rast (dict): Checkerboard raster data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    width = check_rast[cell_nb]["repeated_sequences_times"][0][0] / int(1200/2)
    seq_length = (check_rast[cell_nb]["repeated_sequences_times"][0][1] -
                 check_rast[cell_nb]["repeated_sequences_times"][0][0])
    
    ax.bar(np.linspace(0, seq_length, int(1200/2)) + width/2,
           check_rast[cell_nb]["psth"], width=1.3*width)
    
    ax.set_xlabel("Time (s)")
    ax.set_xlim([0, seq_length])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_chirp_stimulus_profile(ax, euler_vec: np.ndarray):
    """Plot chirp stimulus waveform.

    Args:
        ax: Matplotlib axis object
        euler_vec (np.ndarray): Chirp stimulus vector

    Returns:
        None. Modifies axis in place.
    """
    ax.plot(np.linspace(0, 32, 1600), euler_vec[0+151:151+1600, 1], 
            color='k', lw=0.75)
    
    ax.set_yticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 32)


def plot_chirp_raster(ax, Chirp_data: dict, cell_nb: int):
    """Plot raster response to chirp stimulus.

    Args:
        ax: Matplotlib axis object
        Chirp_data (dict): Chirp response data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    ax.eventplot(Chirp_data[cell_nb]["spike_trains"], color='k', alpha=1)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 20)
    ax.set_ylabel("#Trial")
    ax.set_title("Response to the chirp stimulus")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_chirp_psth(ax, Chirp_data: dict, cell_nb: int):
    """Plot PSTH response to chirp stimulus.

    Args:
        ax: Matplotlib axis object
        Chirp_data (dict): Chirp response data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.
    """
    ax.plot(np.linspace(0, 32, 800), Chirp_data[cell_nb]['psth'])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (spikes/s)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 32)


def plot_direction_tuning_polar(ax, DG_data: dict, cell_nb: int):
    """Plot direction/orientation tuning on polar plot.

    Args:
        ax: Matplotlib polar axis object
        DG_data (dict): Direction selectivity data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.

    Raises:
        KeyError: If DG data not available for this cell
    """
    atune = DG_data[cell_nb]['atune']
    R = DG_data[cell_nb]['Rtune']
    TuneSum = DG_data[cell_nb]['Tuning']
    IDX = DG_data[cell_nb]['IDX']
    
    theta = np.linspace(0, 2 * np.pi, 9)
    
    # Plot preferred direction
    ax.plot([atune, atune], [0, R], 'b-')
    ax.plot([atune], [R], 'bo')
    
    # Plot tuning curve
    ax.plot(theta, TuneSum)
    ax.fill(theta, TuneSum, 'b', alpha=0.1)
    
    # Add text annotations
    ax.text(np.pi/2 * 6/8, 2.6, f'IDX = {np.round(IDX, 1)}', size=18)
    ax.text(np.pi/2 * 6/9, 2.2, f'R = {np.round(R, 1)}', size=18)
    
    # Format axis
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels([0, '', 1, '', 2])


def plot_dg_rasters(ax, DG_data: dict, cell_nb: int):
    """Plot rasters for drifting gratings at different orientations.

    Args:
        ax: Matplotlib axis object
        DG_data (dict): Direction selectivity data
        cell_nb (int): Cell ID

    Returns:
        None. Modifies axis in place.

    Raises:
        KeyError: If DG data not available for this cell
    """
    seq_sep = 20
    seq_len = 12
    ch_raster = DG_data[cell_nb]['rasters']
    
    ax.eventplot(ch_raster[:], color='k', lw=1, linelengths=0.95)
    
    # Add orientation separators
    for a in np.arange(8):
        ax.axvline(a*seq_sep, color='gray', lw=2)
        ax.axvline(a*seq_sep + seq_len, color='gray', lw=2)
        ax.axvline(a*seq_sep + seq_len/6, color='gray', ls='--', lw=1.5)
    
    ax.set_xlim([-seq_sep/2, seq_sep*8])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([6, 26, 46, 66, 86, 106, 126, 146], 
                  [0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_title('DG rasters')


# =============================================================================
# ID CARD MAIN FUNCTION
# =============================================================================

def create_single_id_card(fig, gs, cell_nb: int, exp: str, cluster,
                         cell_rpvs: dict, sta_results: dict, check_rast: dict,
                         euler_vec: np.ndarray, Chirp_data: dict, DG_data: dict,
                         rpv_len: float = 2.0):
    """Create complete ID card for a single cell.

    Args:
        fig: Matplotlib figure object
        gs: GridSpec object for subplot layout
        cell_nb (int): Cell ID
        exp (str): Experiment name
        cluster: Cluster assignment (int or 'Not assigned')
        cell_rpvs (dict): Refractory period violation data
        sta_results (dict): STA analysis results
        check_rast (dict): Checkerboard raster data
        euler_vec (np.ndarray): Chirp stimulus vector
        Chirp_data (dict): Chirp response data
        DG_data (dict): Direction selectivity data
        rpv_len (float): Refractory period length in ms. Default 2.0.

    Returns:
        None. Modifies figure in place.

    Note:
        Creates comprehensive visualization including:
        - ISI histogram
        - Spatial and temporal STAs
        - Checkerboard responses
        - Chirp responses
        - Direction tuning (if available)
    """
    # Set title
    plt.suptitle(f"exp{exp} _c{cell_nb}  - Cluster_group_{cluster}", fontsize=20)
    
    # Plot ISI histogram
    ax = fig.add_subplot(gs[0:2, 0:1])
    plot_isi_histogram(ax, cell_rpvs, cell_nb, rpv_len)
    
    # Plot spatial STA
    ax = fig.add_subplot(gs[0:2, 3:5])
    plot_spatial_sta(ax, sta_results, cell_nb)
    
    # Plot temporal STA
    ax = fig.add_subplot(gs[0:2, 1:3])
    plot_temporal_sta(ax, sta_results, cell_nb)
    
    # Plot chirp stimulus profile
    ax = fig.add_subplot(gs[3:4, 0:3])
    plot_chirp_stimulus_profile(ax, euler_vec)
    
    # Plot chirp raster
    ax = fig.add_subplot(gs[4:6, 0:3])
    plot_chirp_raster(ax, Chirp_data, cell_nb)
    
    # Plot chirp PSTH
    ax = fig.add_subplot(gs[6:7, 0:3])
    plot_chirp_psth(ax, Chirp_data, cell_nb)
    
    # Plot checkerboard raster
    ax = fig.add_subplot(gs[4:6, 3:5])
    plot_checkerboard_raster(ax, check_rast, cell_nb)
    
    # Plot checkerboard PSTH
    ax = fig.add_subplot(gs[6:7, 3:5])
    plot_checkerboard_psth(ax, check_rast, cell_nb)
    
    # Try to plot direction selectivity data
    try:
        # Plot polar tuning
        ax = fig.add_subplot(gs[2:4, 3:5], polar=True)
        plot_direction_tuning_polar(ax, DG_data, cell_nb)
        
        # Plot DG rasters
        ax = fig.add_subplot(gs[2:3, 0:3])
        plot_dg_rasters(ax, DG_data, cell_nb)
        
    except (KeyError, TypeError, IndexError) as e:
        # DG data not available for this cell - this is okay
        pass


def create_id_cards_and_plots(cells: list, cell_rpvs: dict, output_directory: str,
                              exp: str, sta_results: dict, check_rast: dict,
                              euler_vec: np.ndarray, Chirp_data: dict, DG_data: dict,
                              rpv_len: float = 2.0):
    """Generate and save ID cards for all cells.

    Args:
        cells (list): List of cell IDs to generate ID cards for
        cell_rpvs (dict): Refractory period violation data for all cells
        output_directory (str): Path to save ID cards
        exp (str): Experiment name
        sta_results (dict): STA analysis results
        check_rast (dict): Checkerboard raster data
        euler_vec (np.ndarray): Chirp stimulus vector
        Chirp_data (dict): Chirp response data
        DG_data (dict): Direction selectivity data
        rpv_len (float): Refractory period length in ms. Default 2.0.

    Returns:
        None. Saves ID card figures to output_directory/ID_cards/

    Note:
        Creates comprehensive ID cards showing multiple stimulus responses
        and quality metrics for each cell. Handles missing data gracefully.
    """
    # Create output directory
    fig_directory = os.path.normpath(os.path.join(output_directory, 'ID_cards'))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)
    
    print(f"Creating ID cards for {len(cells)} cells...")
    print(f"Saving to: {fig_directory}")
    
    # Generate ID card for each cell
    for cell_nb in tqdm(cells[:], desc="Generating ID cards"):
        # Create figure with gridspec layout
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(7, 5,
                             left=0.1, right=0.9, bottom=0.1, top=0.9,
                             wspace=0.3, hspace=0.7)
        
        # Get cluster assignment
        if Chirp_data[cell_nb]['type'] == 'Not assigned':
            cluster = ''
        else:
            cluster = Chirp_data[cell_nb]['type']
        
        # Create ID card
        create_single_id_card(
            fig, gs, cell_nb, exp, cluster,
            cell_rpvs, sta_results, check_rast,
            euler_vec, Chirp_data, DG_data, rpv_len
        )
        
        # Save figure
        fsave = os.path.join(fig_directory, f'Group{cluster}_cell{cell_nb}')
        fig.savefig(fsave + '.png', format='png', dpi=110)
        plt.close(fig)
    
    print(f"ID cards saved to: {fig_directory}")


# =============================================================================
# WORKFLOW WRAPPER
# =============================================================================

def run_complete_id_card_workflow(params: dict, rpv_threshold: float = 0.5,
                                  rpv_len: float = 2.0, cells_subset: list = None):
    """Run complete workflow to generate ID cards for all cells.

    Args:
        params (dict): Experiment parameters from params.py
        rpv_threshold (float): RPV threshold for cell filtering. Default 0.5%.
        rpv_len (float): Refractory period length in ms. Default 2.0 ms.
        cells_subset (list or None): Specific cells to process, or None for all cells.

    Returns:
        dict: Dictionary containing:
            - 'good_cells': Array of cells passing quality threshold
            - 'all_cells': List of all cells
            - 'fig_directory': Path to saved ID cards

    Note:
        This workflow:
        1. Loads all necessary data
        2. Filters cells by quality (RPV)
        3. Generates comprehensive ID cards
        Requires cell_rpvs data to be pre-computed and available.
    """
    print("="*60)
    print("STARTING ID CARD GENERATION WORKFLOW")
    print("="*60)
    
    # Step 1: Load all data
    print("\nStep 1: Loading data...")
    (exp, output_directory, phy_directory, check_directory, DG_directory,
     CT_directory, euler_vec, check_rast, DG_data, sta_results, 
     cells, Chirp_data) = import_data_to_plot(params)
    
    print(f"Loaded data for {len(cells)} cells from experiment {exp}")
    
    # Step 2: Load or compute RPV data (assumes this is pre-computed)
    print("\nStep 2: Loading RPV data...")
    # NOTE: This assumes cell_rpvs is already computed and saved
    # You may need to add a function to compute this if not available
    try:
        cell_rpvs = utils.load_obj(os.path.join(output_directory, 'cell_rpvs.pkl'))
    except FileNotFoundError:
        print("WARNING: cell_rpvs.pkl not found. Please compute RPV data first.")
        print("Skipping quality filtering step.")
        cell_rpvs = None
    
    # Step 3: Filter cells by quality
    if cell_rpvs is not None:
        print(f"\nStep 3: Filtering cells (RPV threshold: {rpv_threshold}%)...")
        good_cells = select_and_save_good_cells(
            cells, cell_rpvs, output_directory, rpv_threshold
        )
        print(f"Selected {len(good_cells)}/{len(cells)} cells passing quality threshold")
    else:
        good_cells = np.array(cells)
        print("\nStep 3: Skipped (no RPV data available)")
    
    # Determine which cells to process
    if cells_subset is not None:
        cells_to_process = cells_subset
        print(f"\nProcessing user-specified subset: {len(cells_to_process)} cells")
    else:
        cells_to_process = cells
        print(f"\nProcessing all {len(cells_to_process)} cells")
    
    # Step 4: Generate ID cards
    print("\nStep 4: Generating ID cards...")
    if cell_rpvs is not None:
        create_id_cards_and_plots(
            cells_to_process, cell_rpvs, output_directory,
            exp, sta_results, check_rast, euler_vec, Chirp_data, DG_data, rpv_len
        )
    else:
        print("Cannot generate ID cards without RPV data.")
        return None
    
    fig_directory = os.path.normpath(os.path.join(output_directory, 'ID_cards'))
    
    print("\n" + "="*60)
    print("ID CARD GENERATION COMPLETE!")
    print("="*60)
    print(f"Saved to: {fig_directory}")
    
    return {
        'good_cells': good_cells,
        'all_cells': cells,
        'fig_directory': fig_directory
    }