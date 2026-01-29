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

from utils import extract_from_sequence, load_obj

# LIST OF THINGS MISSING

# - Something to align spike trains and well split repetitions ? (will need to be done using the standard VEC files using the 5th column as sequence + rep ID)


# ==========================
# Loading utilities
# ==========================

### Shouldn't be here, load directly instead
# ==============================================================================
# PSTH + RASTER ANALYSIS
# ==============================================================================

# Important part 
# Some of it is already done in utils.py

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