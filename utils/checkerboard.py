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


#########################################
#####     Checkerboard Analysis     #####
#########################################


def get_recording_spikes(recording_name, all_recs_spikes):
    rec_spikes = {}
    for cell_nb, recordings in list(all_recs_spikes.items()):
        rec_spikes[cell_nb] = recordings[recording_name]
    return rec_spikes


def align_triggers_spikes(triggers, spike_times):
    # Clip the spike times to the recording time
    trigger_start = np.min(triggers)
    trigger_end = np.max(triggers)
    spike_times_filtered = spike_times[
        np.where((spike_times > trigger_start) & (spike_times < trigger_end))
    ]

    # Set trigger & spikes start times to zero
    triggers = triggers - trigger_start
    spike_times_filtered = spike_times_filtered - trigger_start

    return triggers, spike_times_filtered


def build_rasters(
    cell_spikes,
    triggers,
    stim_frequency,
    nb_frames_by_sequence=params.nb_frames_by_sequence,
):
    nb_sequences = int(len(triggers) / nb_frames_by_sequence)
    int(nb_frames_by_sequence / stim_frequency)

    repeated_sequences_times = []
    spike_trains = []
    spikes_counts = np.zeros(int(nb_frames_by_sequence / 2))

    analyse = {}
    for i in range(nb_sequences):
        # Get the repeated sequence times for the specified position
        time_start_id = i * nb_frames_by_sequence + int(nb_frames_by_sequence / 2)
        time_end_id = (i + 1) * nb_frames_by_sequence
        times_sequence = triggers[time_start_id:time_end_id]
        repeated_sequences_times.append((times_sequence[0], times_sequence[-1]))

        # Build the spike trains corresponding to stimulus repetitions & make it start to 0
        spike_sequence = cell_spikes[
            np.where(
                (cell_spikes > repeated_sequences_times[-1][0])
                & (cell_spikes < repeated_sequences_times[-1][1])
            )
        ]
        spike_trains.append(spike_sequence - repeated_sequences_times[-1][0])

        # Compute psth
        spikes_counts += np.histogram(
            spike_trains[-1],
            bins=int(nb_frames_by_sequence / 2),
            range=(
                0,
                repeated_sequences_times[-1][1] - repeated_sequences_times[-1][0],
            ),
        )[0]

    analyse["spike_times"] = cell_spikes
    analyse["repeated_sequences_times"] = repeated_sequences_times
    analyse["spike_trains"] = spike_trains
    analyse["psth"] = (
        spikes_counts / nb_sequences * stim_frequency
    )  # transform spikes_count in mean firing rates
    return analyse


def image_projection(image, mea):
    """
    Project the image following setup transformation of image compared to the bin displayed on a computer before the setup
    image has to be a numpy array. It can have values from 0 to 1 or 0 to 255, both works.
    """
    if mea == 2:
        image = np.rot90(image)
        image = np.flipud(image)

    elif mea == 3:
        image = np.fliplr(image)
    #         image = np.ones(image.shape)*np.max(image)-image  ## Reversing polarity in mea3
    return image


def checkerboard_from_binary(
    nb_frames,
    nb_checks_x,
    nb_checks_y,
    checkerboard_file,
    binary_source_path,
    mea,
):
    binary_source_file = open(binary_source_path, mode="rb")
    checkerboard = np.zeros((nb_frames, nb_checks_x, nb_checks_y), dtype="uint8")

    for frame in tqdm(range(nb_frames)):
        image = np.zeros((nb_checks_x, nb_checks_y), dtype=float)

        for row in range(nb_checks_x):
            for col in range(nb_checks_y):
                bit_nb = (nb_checks_x * nb_checks_y * frame) + (nb_checks_x * row) + col
                binary_source_file.seek(bit_nb // 8)
                byte = int.from_bytes(binary_source_file.read(1), byteorder="big")
                bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)
                if bit == 0:
                    image[row, col] = 0.0
                elif bit == 1:
                    image[row, col] = 1.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)

        checkerboard[frame, :, :] = image_projection(image, mea)
    np.save(checkerboard_file, checkerboard)
    print(f"Checkerboard stimulus created and saved at : {checkerboard_file}")
    return checkerboard


def extract_from_sequence(
    cell_spikes: np.ndarray,
    triggers: np.ndarray,
    nb_repeats: int,
    stim_frequency: float,
    sequence_portion: tuple,
    nb_frames_per_sequence: int,
):
    """
    Extract the spike trains corresponding to the repetitions of a portion of the stimulus sequence and compute spike count and psth for this portion.

    Args:
        cell_spikes (numpy array): Array of spike times for a single cell.
        triggers (numpy array): Array of trigger times corresponding to stimulus frames.
        nb_repeats (int): Number of repetitions of the stimulus sequence.
        stim_frequency (float): Frequency of stimulus presentation in Hz.
        sequence_portion (tuple): Tuple containing the start and end portion of the sequence to analyze (values between 0 and 1).
        nb_frames_per_sequence (int): Total number of frames in one full sequence of the stimulus.

    Returns:
        analyse (dict): Dictionary containing the following keys:
            - "spike_times": Original spike times for the cell.
            - "repeated_sequences_times": List of tuples with start and end times of each repeated sequence portion.
            - "spike_trains": List of numpy arrays, each containing the spike times aligned to the start of the sequence portion for each repetition.
            - "counted_spikes": 2D numpy array of shape (nb_sequences, nb_frames) containing the spike counts for each sequence repetition and time bin.
            - "psth": 1D numpy array containing the mean firing rate across repetitions for each time bin, computed from "counted_spikes".
    """
    nb_sequences = int(len(triggers) / nb_frames_per_sequence)
    int(nb_frames_per_sequence / stim_frequency)

    repeated_sequences_times = []
    spike_trains = []
    spikes_counts = np.zeros((nb_sequences, int(nb_frames_per_sequence / 2)))

    analyse = {}
    for i in range(nb_sequences):
        # Get the repeated sequence times for the specified position
        time_start_id = i * nb_frames_per_sequence + int(
            sequence_portion[0] * nb_frames_per_sequence
        )
        time_end_id = i * nb_frames_per_sequence + int(
            sequence_portion[1] * nb_frames_per_sequence
        )
        times_sequence = triggers[time_start_id:time_end_id]
        repeated_sequences_times.append((times_sequence[0], times_sequence[-1]))

        # Build the spike trains corresponding to stimulus repetitions & make it start to 0
        spike_sequence = cell_spikes[
            np.where(
                (cell_spikes > repeated_sequences_times[-1][0])
                & (cell_spikes < repeated_sequences_times[-1][1])
            )
        ]
        spike_trains.append(spike_sequence - repeated_sequences_times[-1][0])

        # Compute psth
        spikes_counts[i, :] = np.histogram(
            spike_trains[-1],
            bins=int(nb_frames_per_sequence / 2),
            range=(
                0,
                repeated_sequences_times[-1][1] - repeated_sequences_times[-1][0],
            ),
        )[0]

    analyse["spike_times"] = cell_spikes
    analyse["repeated_sequences_times"] = repeated_sequences_times
    analyse["spike_trains"] = spike_trains
    analyse["counted_spikes"] = spikes_counts
    analyse["psth"] = (
        spikes_counts.sum(axis=0) / nb_repeats * stim_frequency
    )  # transform spikes_count in mean firing rates
    return analyse


