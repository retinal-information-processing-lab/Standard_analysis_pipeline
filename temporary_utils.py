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

from utils import extract_from_sequence

# ==========================
# Loading utilities
# ==========================


def prompt_user_for_recording(params: dict, stim_name: str) -> tuple[int, str]:
    """
    Display available recordings and prompt user to select one.
    
    Args:
        params: Dictionary containing experiment parameters with 'recording_names' key

    Returns:
        Selected recording number as integer
        Selected recording name as string
    """
    print(f'Which of the following is the {stim_name}: ')
    for num, rec in enumerate(params.recording_names):
        print(f'\t{num} --> {rec}')

    recording_number = int(input(f'{stim_name} number : '))
    recording_name = params.recording_names[recording_number]
    print(f'Selected recording: {recording_name}\n')
    
    return recording_number, recording_name


def create_analysis_directory(params: dict, recording_number: int, analysis_name: str) -> str:
    """
    Create directory for specified analysis output.
    
    Args:
        params: Dictionary containing 'output_directory' key
        recording_number: Recording number for directory naming
        analysis_name: Name of the analysis (e.g., 'DG', 'Checkerboard')
        
    Returns:
        Path to created directory
    """
    check_directory = os.path.normpath(
        os.path.join(params.output_directory, f'{analysis_name}_Analysis_rec_{recording_number}')
    )
    
    if not os.path.isdir(check_directory):
        os.makedirs(check_directory)
    
    return check_directory

def find_Analysis_Directory(dir_type="Checkerboard", output_directory = params.output_directory):
    """
        Automatically calls for the analysis folder using names defined in the pipeline :
            - Checkerboard_Analysis_rec_i
            - DG_Analysis_rec_i
            - CellTyping_Analysis_rec_i
            
        dir_type should be either "Checkerboard", "DG", or "CellTyping"
        
        If severeal analysis has been done for the same type, you will have to input the one to select.
    """
    dirs = sorted([os.path.splitext(f)[0] for f in os.listdir(output_directory) if not (os.path.isfile(os.path.join(output_directory, f))) and dir_type in f])
    if len(dirs)==1:
        analysis_directory = dirs[0]
    elif len(dirs)>1:
        print(f"\n Several {dir_type} analysis folder has been found :")
        print(*['{} : {}'.format(i,dirs[i]) for i, recording_name in enumerate(dirs)], sep="\n")
        analysis_directory = dirs[int(input(f"\n Select the {dir_type} directory to use : "))]
        print(f"\n Selected folder : {analysis_directory} \n")
    else:
        assert len(dirs)>=1, (f"No Directory of type {dir_type} could be found at : \n\t'{output_directory}'\n\nMake sure that you have done the {dir_type} analysis first !")
    
    return os.path.normpath(os.path.join(output_directory,analysis_directory))

def load_triggers(params: dict, rec_name: str) -> tuple[np.ndarray, dict]:
    """
    Load trigger data from saved file and convert indices to seconds.
    
    Args:
        params: Dictionary with 'triggers_directory', 'exp', and 'checkerboard_name'
        rec_name: Selected recording name to load triggers for
        
    Returns:
        Tuple of (triggers array, triggers_data dict)
    """
    triggers_file = os.path.normpath(os.path.join(
        params.triggers_directory,
        f"{params.exp}_{rec_name}_triggers.pkl"
    ))
    triggers_data = load_obj(triggers_file)
    stim_onsets = triggers_data['indices'] / params.fs
    
    return stim_onsets, triggers_data


### Shouldn't be here, load directly instead
def load_spike_trains(params, rec):
    """
    Load spike times for all neurons for a given recording.

    Parameters
    ----------
    params : object
        Experiment parameters.
    rec : str
        Recording name.

    Returns
    -------
    list[np.uint32]
        Cell/cluster identifiers.
    list[np.ndarray]
        Spike times per cell (seconds).
    """
    spike_trains = load_obj(
        os.path.join(params.output_directory,
                     f"{params.exp}_fullexp_neurons_data.pkl")
    )

    cells = list(spike_trains.keys())
    spike_times = [spike_trains[cell][rec] for cell in cells]

    print(f"\nTotal : {len(cells)} neurons loaded")
    print(f"Clusters id :\n{cells}\n")

    return cells, spike_times

# ==============================================================================
# LOAD PIPELINE OUTPUTS
# ==============================================================================

#n USe one loader instead: load_obj

def load_chirp_stimulus_vector(vec_filename: str = "Euler_50Hz_20reps_1024x768pix.vec"):
    """Load chirp stimulus vector for plotting.

    Args:
        vec_filename (str): Filename of the stimulus vector in ./ressources/.
                           Default "Euler_50Hz_20reps_1024x768pix.vec".

    Returns:
        np.ndarray: Chirp stimulus vector with shape (n_frames, n_channels)
    """
    vec_path = os.path.join('./ressources', vec_filename)
    euler_vec = np.genfromtxt(vec_path)
    return euler_vec


def load_checkerboard_rasters(check_directory: str):
    """Load checkerboard raster data.

    Args:
        check_directory (str): Path to checkerboard analysis directory

    Returns:
        dict: Checkerboard raster data with cell IDs as keys
    """
    check_rast = np.load(
        os.path.join(check_directory, 'Check_rasters_data.npy'),
        allow_pickle=True
    ).item()
    return check_rast


def load_direction_selectivity_data(DG_directory: str, exp: str):
    """Load direction/orientation selectivity data.

    Args:
        DG_directory (str): Path to direction selectivity analysis directory
        exp (str): Experiment name

    Returns:
        dict: Direction selectivity data with cell IDs as keys
    """
    DG_data = np.load(
        os.path.join(DG_directory, f'DG_data_exp{exp}.pkl'),
        allow_pickle=True
    )
    return DG_data


def load_sta_results(check_directory: str):
    """Load STA (spike-triggered average) analysis results.

    Args:
        check_directory (str): Path to checkerboard analysis directory

    Returns:
        tuple: Contains:
            - sta_results (dict): STA analysis results with cell IDs as keys
            - cells (list): List of cell IDs
    """
    sta_results = np.load(
        os.path.join(check_directory, 'sta_data_3D_fitted.pkl'),
        allow_pickle=True
    )
    cells = list(sta_results.keys())
    return sta_results, cells


def load_chirp_data(CT_directory: str, exp: str):
    """Load chirp stimulus response data.

    Args:
        CT_directory (str): Path to cell typing analysis directory
        exp (str): Experiment name

    Returns:
        dict: Chirp response data with cell IDs as keys
    """
    Chirp_data = np.load(
        os.path.join(CT_directory, f'{exp}_cell_typing_data.pkl'),
        allow_pickle=True
    )
    return Chirp_data

# ==============================================================================
# RASTER ANALYSIS
# ==============================================================================

def compute_rasters(spikes: dict, triggers: np.ndarray, 
                   nb_repeats: int, stimulus_frequency: int) -> dict:
    """
    Compute raster data for all cells.
    
    NOTE : Redundant with build_rasters
    
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
    
    for cell_id, spike_times in tqdm(spikes.items()):
        raster_data[cell_id] = extract_from_sequence(
            spike_times, triggers, nb_repeats, stim_frequency=stimulus_frequency
        )
    
    return raster_data

def plot_single_raster(ax, spike_trains, color='darkblue', linelength=0.8):
    """
    Plot raster for a single cell.
    From data directly.

    Args:
        ax: Matplotlib axis 
        spike_trains: List of spike trains
    """
    ax.eventplot(spike_trains, colors=color, linelengths=linelength)
    ax.set(title="Raster plot", ylabel="N Repetitions", xlabel="Time in sec",)

def plot_single_psth_from_raster_data(ax, raster_data, cell_nb, params: dict):
    """
    Plot PSTH for a single cell.
    From raster data directly to extract all information automatically.

    Args:
        ax: Matplotlib axis
        raster_data: Dictionary with raster data
        cell_nb: Cell number to plot
        params: Dictionary with 'nb_frames_by_sequence'
    """
    width = raster_data[cell_nb]["repeated_sequences_times"][0][0] / int(
        params.nb_frames_by_sequence / 2
    )
    seq_length = (
        raster_data[cell_nb]["repeated_sequences_times"][0][1]
        - raster_data[cell_nb]["repeated_sequences_times"][0][0]
    )

    x_vals = (
        np.linspace(0, seq_length, int(params.nb_frames_by_sequence / 2)) + width / 2
    )
    ax.bar(x_vals, raster_data[cell_nb]["psth"], width=1.3 * width)
    ax.set(xlabel="Time in sec", ylabel="Firing rate (spikes/s)")