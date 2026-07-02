import numpy as np
import pickle
import os
import gc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
from scipy.signal import convolve
from skimage import measure
import itertools
import time
from collections import defaultdict
import matplotlib.path as mpltPath
from types import ModuleType

import params

from .preprocessing import *  # load_obj, save_obj

# ==========================
# Baptiste testedd utils => To move in the right area
# ==========================


def load_spike_times(rec, output_directory, exp, verbose=False):
    """
    Load spike times for all neurons for a given recording from the fullexp_neurons_data.pkl file.

    Parameters
    ----------
    rec : str
        Recording name.
    output_directory : str
        Analysis output directory (holds ``<exp>_fullexp_neurons_data.pkl``).
    exp : str
        Experiment name.

    Returns
    -------
    list[np.uint32]
        Cell/cluster identifiers.
    dict[int, np.ndarray]
        Spike times per cell (seconds).
    """
    spike_trains = load_obj(
        os.path.join(output_directory, f"{exp}_fullexp_neurons_data.pkl")
    )

    cells = list(spike_trains.keys())
    spike_times = {cell: spike_trains[cell][rec] for cell in cells}
    if verbose:
        print(f"\nTotal : {len(cells)} neurons loaded")
        print(f"Clusters id :\n{cells}\n")

    return cells, spike_times


def load_stim_onset_from_triggers_path(
    triggers_path: str, fs: float, verbose: bool = False
) -> np.ndarray:
    """
    Load trigger data from saved file and give the stim onset already converted in second.

    Args:
        triggers_path: Path to the saved trigger data file (e.g., "../triggers_data.pkl").
        fs: Sampling rate of the MEA in Hz (params.fs).
        verbose: If True, print information about the loaded triggers.

    Returns:
        stim_onsets: Numpy array of stimulus onset times in seconds.
    """
    triggers_data = load_obj(triggers_path)
    stim_onsets = triggers_data["indices"] / fs
    if verbose:
        print(f"Total triggers number : {len(stim_onsets)}")
        print(f"Triggers type loaded : {triggers_data['trigger_type']}")

    return stim_onsets


def prompt_user_for_recording(recording_names, stim_name: str) -> tuple[int, str]:
    """
    Display available recordings and prompt user to select one.

    Args:
        recording_names: List of recording names to choose from (params.recording_names)
        stim_name: Name of the stimulus type (e.g., "Checkerboard", "DG") to display in the prompt

    Returns:
        Selected recording number as integer
        Selected recording name as string
    """
    print(f"Which of the following is the {stim_name}: ")
    for num, rec in enumerate(recording_names):
        print(f"\t{num} --> {rec}")

    recording_number = int(input(f"{stim_name} number : "))
    recording_name = recording_names[recording_number]
    print(f"Selected recording: {recording_name}\n")

    return recording_number, recording_name


def create_analysis_directory(
    output_directory: str, recording_number: int, analysis_name: str
) -> str:
    """
    Create directory for specified analysis output.

    Args:
        output_directory: Base directory where the analysis folder is created (params.output_directory)
        recording_number: Recording number for directory naming
        analysis_name: Name of the analysis (e.g., 'DG', 'Checkerboard')

    Returns:
        Path to created directory
    """
    print("Creating analysis directory...")
    print(output_directory)
    check_directory = os.path.normpath(
        os.path.join(
            output_directory, f"{analysis_name}_Analysis_rec_{recording_number}"
        )
    )

    if not os.path.isdir(check_directory):
        os.makedirs(check_directory)

    return check_directory


def find_analysis_directory(output_directory, dir_type="Checkerboard"):
    """
    Automatically calls for the analysis folder using names defined in the pipeline :
        - Checkerboard_Analysis_rec_i
        - DG_Analysis_rec_i
        - CellTyping_Analysis_rec_i

    dir_type should be either "Checkerboard", "DG", or "CellTyping"

    If severeal analysis has been done for the same type, you will have to input the one to select.
    """
    dirs = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(output_directory)
            if not (os.path.isfile(os.path.join(output_directory, f))) and dir_type in f
        ]
    )
    if len(dirs) == 1:
        analysis_directory = dirs[0]
    elif len(dirs) > 1:
        print(f"\n Several {dir_type} analysis folder has been found :")
        print(
            *["{} : {}".format(i, dirs[i]) for i, recording_name in enumerate(dirs)],
            sep="\n",
        )
        analysis_directory = dirs[
            int(input(f"\n Select the {dir_type} directory to use : "))
        ]
        print(f"\n Selected folder : {analysis_directory} \n")
    else:
        assert (
            len(dirs) >= 1
        ), f"No Directory of type {dir_type} could be found at : \n\t'{output_directory}'\n\nMake sure that you have done the {dir_type} analysis first !"

    return os.path.normpath(os.path.join(output_directory, analysis_directory))


