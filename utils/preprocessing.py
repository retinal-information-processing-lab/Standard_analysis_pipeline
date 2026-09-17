import numpy as np
import pickle
import os
from tqdm.auto import tqdm
import csv
from colorama import Fore, Style

import params
from pathlib import Path

#############################################
######          Preprocessing          ######
#############################################


def create_symlinks(
    recording_names,
    symbolic_link_directory=params.symbolic_link_directory,
    recording_directory=params.recording_directory,
    print_warning=True,
):
    """
    Function to create symbolic links to the recording files for spyking circus needs.

    Input :
        - recording_names (list) : Each item is a recording raw file name
        - symbolic_link_directory (string) : Path to the folder where the symbolic links must be saved
        - recording_directory (string) : Path to the folder containing the raw records

    Output :
        - linknames_list (list) : list of stings of the symbolic links needed for the sorting
        - previously_existing (list) : list of strings (can be called as bool aswell) reporting if the file existed before or not


    Possible mistakes :
        - Permission to write symbolic links denied, restart the notebook from an administrator shell
        - Wrong folders
    """
    linknames_list = []
    previously_existing = []

    for i_recording, filename in enumerate(recording_names):
        linkname = "recording_{}.raw".format(
            str(i_recording).zfill(2)
        )  # Create this iteration link name following spyking circus expected raw files names format (recording_ii.raw)
        linknames_list.append(
            linkname
        )  # linknames_list is created with the extention in the names
        if os.path.exists(
            os.path.join(symbolic_link_directory, linkname)
        ):  # Check if the symbolic link exists already at given path for this indice
            if print_warning:
                print(
                    Fore.YELLOW
                    + r"/!\ File {} already exists /!\ ".format(
                        os.path.join(symbolic_link_directory, linkname)
                    )
                    + Style.RESET_ALL
                )
                print(
                    Fore.YELLOW
                    + "\t\tMay not be a problem if you already run this code for THIS experiment\n"
                    + Style.RESET_ALL
                )
            previously_existing.append(" already existed")
            continue  # If yes, add 'already exists' to previously_existing list and go to next file iteration without rewriting current trig data
        try:
            os.symlink(
                os.path.join("../" + os.path.split(recording_directory)[1], filename),
                os.path.join(
                    "../" + os.path.split(symbolic_link_directory)[1], linkname
                ),
            )
            previously_existing.append(
                ""
            )  # If no, create symlink accordinly and add an empty string to 'previously_existing' list
        except FileExistsError:
            raise FileExistsError(
                r"/!\ Old missmatching SymLinks already in your sorting folder. Delete them and retry ! /!\ path : {}".format(
                    os.path.join(symbolic_link_directory, linkname)
                )
            )
    return linknames_list, previously_existing
    # Return both link names created and the tracking of previously existing links




def load_trigger_channels(
    input_path,
    channel_ids=None,
    dtype=params.dtype,
    nb_channels=params.nb_channels,
    voltage_resolution=params.voltage_resolution,
):
    """
    Read the trigger / auxiliary channels of a raw recording, in physical microvolts.

    A few MEA channels carry experiment metadata rather than neural signal (the stimulus
    trigger and auxiliary channels: holographic trigger, shutter state, colour code...).
    This reads the ones listed in ``channel_ids`` and returns their raw traces, so they can
    be saved for later sanity checks — always, without needing to know the recording type.

    Unlike load_data (which reads a single channel with a per-sample loop), this reads every
    requested channel in one vectorised pass, so reading several channels is cheap. The
    traces are returned as float32 to keep the saved file half the size of float64.

    Input :
        - input_path (str) : path to the raw binary recording
        - channel_ids (dict) : {name: channel_id} of the channels to read; defaults to
            params.trigger_channel_ids. Entries whose id is None are skipped.
        - dtype (str) : raw data type
        - nb_channels (int) : total number of channels on the mea
        - voltage_resolution (float) : µV per ADC level for this rig (see load_data)

    Output :
        - channels (dict) : {name: 1D float32 numpy array of µV} for every channel whose id
            is not None. Empty dict if nothing to read.

    Possible mistakes :
        - Wrong channel indices in params.trigger_channel_ids
        - nb_channels inconsistent with the file (raises)
    """
    if channel_ids is None:
        channel_ids = params.trigger_channel_ids
    wanted = {name: ch for name, ch in channel_ids.items() if ch is not None}
    if not wanted:
        return {}

    m = np.memmap(os.path.normpath(input_path), dtype=dtype)
    if m.size % nb_channels != 0:
        raise Exception("number of channels is inconsistent with the data size.")
    samples = m.reshape(-1, nb_channels)  # (nb_samples, nb_channels) view

    offset = np.iinfo("int16").min
    channels = {}
    for name, channel_id in wanted.items():
        # Same conversion as load_data: uint16 offset-binary -> signed level -> µV.
        # Computed in float64 for accuracy, stored as float32 to halve the file size.
        trace = samples[:, channel_id].astype(float)
        channels[name] = ((trace + offset) * voltage_resolution).astype(np.float32)
    return channels


def is_holographic_rec(
    input_path, probe_size=params.fs * params.time, mea=params.MEA, dtype=params.dtype
):
    """
    Function to check if a recording was holographic or not

    Input :
        - input_path (str) : path to binary file
        - probe_size (int) : read only part of the recording to reduce useless computation time (default 10s)


    Output :
        - (bool) : if true, the recording is considered as holo because the holo trigger channel has enough data to be considered as active

    Possible mistakes :
        - Wrong folders/files names
        - params.py mea value not on the right rig
        - probe_size has been change and threshold of detection must be ajusted to the new probe_size value to detect holo stims correctly
    """
    #     print('Checking if holographic recording...\t',  end ='')
    if mea == 3:
        return (
            load_data(
                input_path=input_path,
                channel_id=params.holo_channel_id,
                probe_size=probe_size,
                disable=True,
            )[0].max()
            > 0
        )
    else:
        return False



def detect_offsets(data, threshold=params.threshold):
    """
    Function to compute time point in the data coresponding to the shutdown of laser, laser offset trigger

    Input :
        - data (1D numpy array) : raw triggers data
        - threshold (int) : voltage value that detects onsets in data

    Output :
        - indices (1D numpy array) : list of time indices corresponding to the detected offsets time point

    Possible mistakes :
        - Threshold is no longer optimum and has to be changed
        - Wrong mea given as parameters
        - Data coming from the wrong channel
    """

    test_1 = data[:-1] > threshold
    test_2 = data[1:] <= threshold
    test = np.logical_and(test_1, test_2)

    indices = np.where(test)[0]

    test = data[indices - 1] < data[indices]
    while np.any(test):
        indices[test] = indices[test] - 1
        test = data[indices - 1] < data[indices]

    return indices


def write_dead_times_file(
    triggers_list,
    onsets,
    output_directory,
    exp=params.exp,
    time_before=params.time_before,
    time_after=params.time_after,
    offset_time=params.offset_time,
    fs=params.fs,
):
    """
        Create a file called "{experiment_name}_dead_times.dead" containing the dead periods (in ms) to exclude from the analysis for spyking circus purposes

    Input :
        - triggers_list (list) : Ordered list of several records triggers (list of list of triggers)
        - onsets (list) : Ordered list of each of files onsets on wich perform the dead time processing
        - exp (str) : experiment name
        - time_before (int) : Time in ms before a trigger to remove
        - time_after (int) : Time in ms after a trigger to remove
        - offset_time (float) : Time in s after a trigger to add a second virtual trigger to be processed in the dead times
        - fs (int) : sampling rate of the mea in time points per sec

    Possible mistakes :
        - triggers_list doesn't have the right shape (list of list of spikes)
        - onset dictionnary is given instead of a list of the onsets time
        - Permission error, restart from an admin shell
    """

    with open(
        os.path.join(output_directory, "{}_dead_times.dead".format(exp)), "w"
    ) as f:
        if len(triggers_list) != len(onsets):
            print(
                "Onsets list and triggers_list must be the same length. triggers_list must contain a list of reconding triggers not directly the triggers!"
            )
            raise

        for i in range(len(triggers_list)):
            triggers_s = triggers_list[i]
            triggers_s += onsets[i]
            triggers_s = triggers_s.astype("float64") / fs

            if offset_time > 0:
                offset_triggers = np.zeros(len(triggers_s))
                for i, trigger in enumerate(triggers_s):
                    offset_triggers[i] = trigger + offset_time
                triggers_s = np.append(triggers_s, offset_triggers)
                triggers_s = np.sort(triggers_s)
            triggers_ms = triggers_s * 1000  # Convert triggers from s to ms
            for trigger in triggers_ms:
                f.write("{} {}\n".format(trigger - time_before, trigger + time_after))




