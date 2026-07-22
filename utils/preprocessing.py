import numpy as np
import pickle
import os
from tqdm.auto import tqdm
import csv
from colorama import Fore, Style

import params


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


def load_data(
    input_path,
    dtype=params.dtype,
    nb_channels=params.nb_channels,
    channel_id=params.visual_channel_id,
    probe_size=None,
    voltage_resolution=params.voltage_resolution,
    disable=False,
):
    """
    Function to load raw binary file for a given channel signal

    Input :
        - input_path (str) : path to binary file
        - dtype (str) : type of data size
        - nb_channels (int) : total number of channels on the mea
        - channel_id (int) : channel number to be read
        - probe_size (int) : if not None, read only part of the recording (used to check a channel did record some signal)
        - voltage_resolution (float) : µV per ADC level for this rig (derived in params from
            the amplifier gain and input range; see params.voltage_resolution_uV)
        - disable (bool) : True to disable tqdm loading bar

    Output :
        - data (1D numpy array) : signal of the read channel, in physical microvolts (µV)
        - nb_samples (int) : number of time points in the recording

    Possible mistakes :
        - File doesn't exists, check input_path and folders
        - Type error due to very long recording exceeding dtype capcities
    """

    # Load data.
    m = np.memmap(os.path.normpath(input_path), dtype=dtype)

    # Input file sanity check
    if m.size % nb_channels != 0:
        message = "number of channels is inconsistent with the data size."
        raise Exception(message)

    nb_samples = m.size // nb_channels

    if probe_size:
        nb_samples = min(probe_size, nb_samples)

    data = np.empty((nb_samples,), dtype=dtype)
    for k in tqdm(range(nb_samples), disable=disable):
        data[k] = m[nb_channels * k + channel_id]
    data = data.astype(float)
    data = data + np.iinfo("int16").min  # uint16 offset-binary -> signed ADC level
    data = data * voltage_resolution  # ADC level -> physical microvolts (µV)

    return data, nb_samples


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


def detect_onsets(data, threshold=params.threshold):
    """
    Function to compute time point in the data coresponding to the display of a new frame of the stimuli based on trigger recording

    Input :
        - data (1D numpy array) : raw triggers data
        - threshold (int) : voltage value that detects onsets in data

    Output :
        - indices (1D numpy array) : list of time indices corresponding to the detected onsets time point

    Possible mistakes :
        - Threshold is no longer optimum and has to be changed
        - Wrong mea given as parameters
        - Data coming from the wrong channel
    """
    test_1 = data[:-1] < threshold
    test_2 = data[1:] >= threshold
    test = np.logical_and(test_1, test_2)

    indices = np.where(test)[0]

    test = data[indices - 1] < data[indices]
    while np.any(test):
        indices[test] = indices[test] - 1
        test = data[indices - 1] < data[indices]

    return indices


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


def save_obj(obj, name):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """

    if os.path.dirname(os.path.normpath(name)) != "":
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(), os.path.normpath(name))

    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(os.path.normpath(name), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
        Generic function to load a bin obj with pickle protocol

    Input :
        - name (str) : path to where the obj is
    Output :
        - (python object) : loaded object

    Possible mistakes :
        - Wrong path
    """

    if os.path.dirname(os.path.normpath(name)) != "":
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(), os.path.normpath(name))
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(os.path.normpath(name), "rb") as f:
        return pickle.load(f)


def recording_onsets(
    recording_names,
    path=params.recording_directory,
    nb_bytes_by_datapoint=params.nb_bytes_by_datapoint,
    nb_channels=params.nb_channels,
):
    """
        Read from raw files (either links or recondings) the onsets for each rec

    Input :
        - recording_names (list) : Ordered list of raw files names to open and read length
        - path (str) : path to the directory containing the files
        - nb_bytes_by_datapoint (int) : size in byte of each time points
        - nb_channels (int) : number of channels of the mea
    Output :
        - onsets (dict) : Dictionnary of all onsets using recording_names as dict key

    Possible mistakes :
        - Wrong folders given as input
        - Mea number is wrong
    """

    onsets = {}

    # The onset of the first recording is set to 0
    cursor = 0
    for rec in recording_names:
        onsets[rec] = cursor
        if rec[-4:] == ".raw":
            file_stats = os.stat(os.path.normpath(os.path.join(path, rec)))
        else:
            file_stats = os.stat(os.path.normpath(os.path.join(path, rec + ".raw")))
        cursor += int(file_stats.st_size / (nb_bytes_by_datapoint * nb_channels))
    onsets["end"] = cursor
    return onsets


def run_minimal_sanity_check(
    triggers,
    sampling_rate=params.fs,
    maximal_jitter=params.maximal_jitter,
    stim_type="visual",
):
    """
        Compare the duration of each frame (ie distance between triggers) to see if a max error is reached

    Input :
        - triggers (list) : list of time point of triggers
        - sampling_rate (int) : number of time points per sec
        - maximal_jitter (int) : maximal error admissible in sec
        - stim_type (str) : stimulus type (ie 'visual' or 'holo') used to avoid doing sanity checks on holo triggers
    Output :
        - (1D numpy array) : Array of triggers time points that violate the maximum error

    Possible mistakes :
        - Stim_type given is wrong and holo stim type is given, check the calling of the function
        - indices threshold is wrong and some triggers are missed
        - triggers are corrupted
        - maximal_jitter is too restrictive
    """
    if stim_type == "holo":
        print("No sanity checks done on holographic stimulus")
        return np.array([]).astype("int64")
    elif len(triggers) < 2:
        print(
            "No sanity check performed, only 1 trigger detected. Is threshold correct ?"
        )
        return np.array([]).astype("int64")

    # Check trigger statistics.
    inter_triggers = np.diff(triggers)
    inter_trigger_values, inter_trigger_counts = np.unique(
        inter_triggers, return_counts=True
    )

    index = np.argmax(inter_trigger_counts)
    inter_trigger_value = inter_trigger_values[index]
    errors = np.where(
        np.abs(inter_triggers - inter_trigger_value) >= maximal_jitter * sampling_rate
    )[0]

    if errors.size > 0:
        print(
            r"Minimal sanity checks :\t/!\ Triggers are not evenly spaced /!\ \nNumber of errors : {}\nMaximum error : {} sampling points compared to {} sampling points per trigger".format(
                len(errors), max(np.abs(inter_trigger_values)), inter_trigger_value
            )
        )
    else:
        print("Minimal sanity checks : Ok on all {} triggers".format(len(triggers)))

    return triggers[errors].astype("int64")


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


def extract_all_spike_times_from_phy(directory):
    """
        Read phy variables and extract the spiking times of each cluster
    Input :
        - directory (str) : phy varariables directory
    Output :
        - spike_times (dict) : Dictionnary of each cluster's spiking time, cluster_id as key and a list as value

    Possible mistakes :
        - Wrong directory
        - .npy files no longer exists

    """
    path_all_spike_clusters = os.path.join(directory, "spike_clusters.npy")
    if os.path.isfile(path_all_spike_clusters):
        all_spike_clusters = np.load(path_all_spike_clusters)
    else:
        path_all_spike_clusters = os.path.join(directory, "spike_templates.npy")
        all_spike_clusters = np.load(path_all_spike_clusters)

    all_spike_times = np.load(os.path.join(directory, "spike_times.npy"))

    spike_times = {}
    for i in tqdm(range(len(all_spike_times))):
        if all_spike_clusters[i] not in spike_times.keys():
            spike_times[all_spike_clusters[i]] = []
        spike_times[all_spike_clusters[i]] += [all_spike_times[i]]

    return spike_times


def extract_cluster_groups(phy_path=params.phy_directory):
    """
        Read phy variables and extract the cluster numbers and their group
    Input :
        - phy_path (str) : phy varariables directory
    Output :
        - cluster_number (list) : list of all cluster numbers available in phy
        - good_clusters (list) : subset of previous list containing only clusters labeled as 'good' (ie not noise or mua)

    Possible mistakes :
        - Wrong directory
        - .tsv files no longer exists
    """
    cluster_number = []
    good_clusters = []
    path_cluster_group = os.path.normpath(os.path.join(phy_path, "cluster_group.tsv"))
    path_spike_clusters = os.path.normpath(os.path.join(phy_path, "spike_clusters.npy"))
    path_spike_templates = os.path.normpath(
        os.path.join(phy_path, "spike_templates.npy")
    )

    if os.path.isfile(path_cluster_group):
        print("Extracting Manually Curated 'Good' clusters")
        cluster_file = open(path_cluster_group)
        read_file = csv.reader(cluster_file, delimiter="\t")
        next(cluster_file, None)

        for row in read_file:
            cluster_number += [int(row[0])]
            if row[1] == "good":
                good_clusters += [int(row[0])]

    elif os.path.isfile(path_spike_clusters):
        print(
            "Manual curation not done yet. Extracting all clusters using 'spike_clusters.npy'!"
        )
        spikes_clusters = np.load(path_spike_clusters)
        cluster_number = set(spikes_clusters)
        good_clusters = set(spikes_clusters)

    elif os.path.isfile(path_spike_templates):
        print(
            "Phy hasn't been opened. Extracting all clusters using 'spike_templates.npy'!"
        )
        spikes_templates = np.load(path_spike_templates)
        cluster_number = set(spikes_templates)
        good_clusters = set(spikes_templates)
    else:
        print("No phy files could be opened...\n\n")

    return cluster_number, good_clusters


def split_spikes_by_recording(all_spike_times, good_clusters, onsets, fs=params.fs):
    """
        Function to order all spikes by cluster and recording
    Input :
        - all_spike_times (dict) : cluster number as key and list of spyking times in list as value
        - good_clusters (list) : list of all good clusters id
        - onsets (dict) : key recondings names with onset of the recording as value
    Output :
        - data (data) : dictionnary using cluster_id as key and a second dictionnary as value, this second uses recording names as key and a list of spiking times in sec as value.
                        Call like this data[cluster_id][recording_name]

    Possible mistakes :
        - Given var don't follow the right shape (see above)
    """
    data = {}
    for cell_nb in tqdm(good_clusters):
        data[cell_nb] = {}

        is_first_iteration = True
        for rec, onset in onsets.items():
            if is_first_iteration:
                rec_name = rec
                recording_start_time = onset
                is_first_iteration = False
                continue

            recording_end_time = onset
            rec_spikes = np.array(all_spike_times[cell_nb])[
                (np.array(all_spike_times[cell_nb]) > recording_start_time)
                & (np.array(all_spike_times[cell_nb]) < recording_end_time)
            ]
            data[cell_nb][rec_name] = (rec_spikes - recording_start_time) / fs

            recording_start_time = onset
            rec_name = rec
    return data
