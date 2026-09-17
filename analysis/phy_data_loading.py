import os
import numpy as np
from tqdm import tqdm
import csv
from analysis_params import recording_params, dmd_channel, dmd_threshold
from analysis.tools import save_obj

def recording_onsets(
    recording_names,
    path,
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
        cursor += int(file_stats.st_size / (recording_params['nb_bytes_by_datapoint'] * recording_params['nb_channels']))
    onsets["end"] = cursor
    return onsets


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


def split_spikes_by_recording(all_spike_times, good_clusters, onsets):
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
            data[cell_nb][rec_name] = (rec_spikes - recording_start_time) / recording_params['fs']

            recording_start_time = onset
            rec_name = rec
    return data


def extract_cluster_groups(phy_path):
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


def load_data(
    input_path,
    dtype,
    nb_channels,
    channel_id,
    probe_size,
    voltage_resolution,
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



def detect_onsets(data, threshold, fig_savename=None):
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

    # import matplotlib.pyplot as plt
    # n_pts = int(data.size / 5)
    # # Plot the data
    # plt.figure(figsize=(12, 5))
    #
    # plt.plot(data, label="data", color="steelblue")
    # plt.axhline(threshold, color="red", linestyle="--", label="threshold")
    #
    #
    # plt.xlim(0, n_pts)
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.grid(alpha=0.25)
    # plt.savefig(fig_savename, dpi=300, bbox_inches="tight")
    return indices


def run_minimal_sanity_check(
    triggers,
    sampling_rate,
    maximal_jitter,
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

    if len(triggers) < 2:
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


def extract_phy_data(raw_dir, phy_dir, output_dir, sid, overwrite=False):

    # Extract spikes
    recording_names = [f.name for f in raw_dir.iterdir() if f.suffix == '.raw']
    rec_onsets = recording_onsets(recording_names, path=raw_dir)
    cluster_number, good_clusters = extract_cluster_groups(phy_dir)
    print(f"{len(good_clusters)} good clusters ({len(cluster_number)} total)\n")
    if len(good_clusters) < 1:
        return False
    print("Extracting spike times from phy...")
    all_spike_times = extract_all_spike_times_from_phy(phy_dir)

    print("Splitting spikes per recording, per neuron...")
    good_data = split_spikes_by_recording(all_spike_times, good_clusters, rec_onsets)

    save_name = output_dir / f'{sid}_fullexp_neurons_data.pkl'

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    save_obj(good_data, save_name)
    print(f"\nSaved: {save_name}")


    #Extract triggers
    # from utils import run_minimal_sanity_check

    for rec_name in recording_names:
        if 'checkerboard' not in rec_name:
            continue
        print(f"\n----- Triggers {rec_name}) -----")

        input_file = raw_dir / rec_name
        trigger_out_file = output_dir / f"{sid}_{rec_name}_triggers.pkl"
        if trigger_out_file.exists() and not overwrite:
            continue
        data_out_file = output_dir / f"{sid}_{rec_name}_triggers_data.pkl"

        # Visual stimulus -> triggers are on the visual channel (no holography here).
        data, t_tot = load_data(
            input_path=input_file,
            dtype=recording_params['dtype'],
            nb_channels=recording_params['nb_channels'],
            channel_id=255,
            probe_size=None,
            voltage_resolution=recording_params['data_voltage_resolution'],
        )

        indices = detect_onsets(data, dmd_threshold)
        if len(indices) == 0:
            return False
        # # assert len(indices) > 0, f'NO TRIGGERS FOUND {sid}'
        # # indices_errors = run_minimal_sanity_check(indices, sampling_rate=params.fs, maximal_jitter=params.maximal_jitter,)
        #
        # n = len(data) // 3
        # data_5th = data[:n]
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 4))
        # plt.plot(data_5th, label="data")
        # plt.axhline(dmd_threshold, color="red", linestyle="--", label="dmd_threshold")
        #
        # plt.xlabel("Sample")
        # plt.ylabel("Value")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


        save_obj(
            {
                "indices": indices,
                "duration": t_tot,
                "trigger_type": 'dmd',
                "indice_errors": None,
            },
            trigger_out_file,
        )
        save_obj(data, data_out_file)

    return True
