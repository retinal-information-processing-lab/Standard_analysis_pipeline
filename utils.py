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

#############################################
######          Preprocessing          ######
#############################################




def create_symlinks(recording_names, symbolic_link_directory=params.symbolic_link_directory, recording_directory=params.recording_directory, print_warning=True):
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
        linkname = "recording_{}.raw".format(str(i_recording).zfill(2))                                       #Create this iteration link name following spyking circus expected raw files names format (recording_ii.raw)
        linknames_list.append(linkname)                                                        #linknames_list is created with the extention in the names
        if os.path.exists(os.path.join(symbolic_link_directory,linkname)):                      #Check if the symbolic link exists already at given path for this indice
            if print_warning:
                print(Fore.YELLOW+r"/!\ File {} already exists /!\ ".format(os.path.join(symbolic_link_directory,linkname))+Style.RESET_ALL)
                print(Fore.YELLOW+"\t\tMay not be a problem if you already run this code for THIS experiment\n"+Style.RESET_ALL)
            previously_existing.append(' already existed')
            continue                                                                                #If yes, add 'already exists' to previously_existing list and go to next file iteration without rewriting current trig data
        try:
            os.symlink(os.path.join("../"+os.path.split(recording_directory)[1], filename), os.path.join("../"+os.path.split(symbolic_link_directory)[1],linkname))
            previously_existing.append('')  #If no, create symlink accordinly and add an empty string to 'previously_existing' list
        except FileExistsError as e:
            raise FileExistsError(r"/!\ Old missmatching SymLinks already in your sorting folder. Delete them and retry ! /!\ ".format(os.path.join(symbolic_link_directory,linkname))) 
    return linknames_list, previously_existing  
                          #Return both link names created and the tracking of previously existing links
        

def load_data(input_path, dtype=params.dtype, nb_channels=params.nb_channels, channel_id=params.visual_channel_id, probe_size=None, voltage_resolution=params.voltage_resolution, disable=False):
    """
    Function to load raw binary file for a given channel signal

    Input :
        - input_path (str) : path to binary file
        - dtype (str) : type of data size
        - nb_channels (int) : total number of channels on the mea
        - channel_id (int) : channel number to be read
        - probe_size (int) : if not None, read only part of the recording (used to check a channel did record some signal)
        - voltage_resolution (float) : voltage step per binary value recorded
        - disable (bool) : True to disable tqdm loading bar
        
    Output :
        - data (1D numpy array) : raw signal of the read channel
        - nb_samples (int) : number of time points in the recording
        
    Possible mistakes :
        - File doesn't exists, check input_path and folders
        - Type error due to very long recording exceeding dtype capcities
    """
    
    # Load data.
    m = np.memmap(os.path.normpath(input_path),dtype=dtype)
    
    #Input file sanity check
    if m.size % nb_channels != 0:
        message = "number of channels is inconsistent with the data size."
        raise Exception(message)
    
    
    nb_samples = m.size // nb_channels
    
    if probe_size:
        nb_samples = min(probe_size,nb_samples)
    
    data = np.empty((nb_samples,), dtype=dtype)
    for k in tqdm(range(nb_samples),disable = disable):
        data[k] = m[nb_channels * k + channel_id]
    data = data.astype(float)
    data = data + np.iinfo('int16').min
    data = data / voltage_resolution
    
    return data, nb_samples


def is_holographic_rec(input_path, probe_size=params.fs*params.time, mea = params.MEA, dtype=params.dtype):
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
        return load_data(input_path=input_path, channel_id=params.holo_channel_id, probe_size=probe_size, disable=True)[0].max()>0
    else :
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

def detect_offsets(data,threshold=params.threshold):
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

def save_obj(obj, name ):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open( os.path.normpath(name), 'wb') as f:
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
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open(os.path.normpath(name), 'rb') as f:
        return pickle.load(f)


def recording_onsets(recording_names, path = params.recording_directory,nb_bytes_by_datapoint = params.nb_bytes_by_datapoint, nb_channels=params.nb_channels):
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
    for rec in recording_names :
        onsets[rec] = cursor
        if rec[-4:] == '.raw':
            file_stats = os.stat(os.path.normpath(os.path.join(path,rec)))
        else :
            file_stats = os.stat(os.path.normpath(os.path.join(path,rec+'.raw')))
        cursor += int(file_stats.st_size/(nb_bytes_by_datapoint*nb_channels))
    onsets['end'] = cursor
    return onsets


def run_minimal_sanity_check(triggers, sampling_rate=params.fs, maximal_jitter=params.maximal_jitter, stim_type = 'visual'):
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
    if stim_type == 'holo' :
        print('No sanity checks done on holographic stimulus')
        return np.array([]).astype('int64')
    elif len(triggers)<2:
        print('No sanity check performed, only 1 trigger detected. Is threshold correct ?')
        return np.array([]).astype('int64')
        
    # Check trigger statistics.
    inter_triggers = np.diff(triggers)
    inter_trigger_values, inter_trigger_counts = np.unique(inter_triggers, return_counts=True)

    index = np.argmax(inter_trigger_counts)
    inter_trigger_value = inter_trigger_values[index]
    errors = np.where(np.abs(inter_triggers - inter_trigger_value) >= maximal_jitter * sampling_rate)[0]
    
    if errors.size>0:
        print(r"Minimal sanity checks :\t/!\ Triggers are not evenly spaced /!\ \nNumber of errors : {}\nMaximum error : {} sampling points compared to {} sampling points per trigger".format(len(errors), max(np.abs(inter_trigger_values)), inter_trigger_value))
    else :
        print("Minimal sanity checks : Ok on all {} triggers".format(len(triggers)))

    return triggers[errors].astype('int64')



def write_dead_times_file(triggers_list, onsets,output_directory, exp = params.exp, time_before = params.time_before, time_after=params.time_after, offset_time=params.offset_time, fs=params.fs):
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
    
    with open(os.path.join(output_directory,"{}_dead_times.dead".format(exp)), "w") as f:
        
        if len(triggers_list) != len(onsets):
            print('Onsets list and triggers_list must be the same length. triggers_list must contain a list of reconding triggers not directly the triggers!')
            raise 
            
        for i in range(len(triggers_list)):
            triggers_s =  triggers_list[i]
            triggers_s += onsets[i]
            triggers_s =  triggers_s.astype('float64')/fs
            
            if offset_time > 0 :
                offset_triggers = np.zeros(len(triggers_s))
                for i, trigger in enumerate(triggers_s):
                    offset_triggers[i] = trigger + offset_time
                triggers_s = np.append(triggers_s, offset_triggers)
                triggers_s = np.sort(triggers_s) 
            triggers_ms = triggers_s * 1000  # Convert triggers from s to ms
            for trigger in triggers_ms:
                f.write("{} {}\n".format(trigger-time_before, trigger+time_after))
                
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

    
def extract_cluster_groups(phy_path = params.phy_directory):
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
    path_cluster_group = os.path.normpath(os.path.join(phy_path,"cluster_group.tsv"))
    path_spike_clusters = os.path.normpath(os.path.join(phy_path,"spike_clusters.npy"))
    path_spike_templates = os.path.normpath(os.path.join(phy_path,"spike_templates.npy"))
    
    if os.path.isfile(path_cluster_group):
        print("Extracting Manually Curated 'Good' clusters")
        cluster_file = open(path_cluster_group)
        read_file = csv.reader(cluster_file, delimiter="\t")
        next(cluster_file, None)

        for row in read_file:
            cluster_number += [int(row[0])]
            if row[1] == 'good':
                good_clusters += [int(row[0])]
    
    elif os.path.isfile(path_spike_clusters):
        print("Manual curation not done yet. Extracting all clusters using 'spike_clusters.npy'!")
        spikes_clusters = np.load(path_spike_clusters)
        cluster_number  = set(spikes_clusters)
        good_clusters   = set(spikes_clusters)
        
    elif os.path.isfile(path_spike_templates):
        print("Phy hasn't been opened. Extracting all clusters using 'spike_templates.npy'!")
        spikes_templates = np.load(path_spike_templates)
        cluster_number   = set(spikes_templates)
        good_clusters    = set(spikes_templates)
    else:
        print("No phy files could be opened...\n\n")

    return cluster_number,good_clusters

def split_spikes_by_recording(all_spike_times, good_clusters, onsets, fs = params.fs):
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
            
            if is_first_iteration : 
                rec_name=rec
                recording_start_time=onset
                is_first_iteration=False
                continue
                
            recording_end_time=onset
            rec_spikes=np.array(all_spike_times[cell_nb])[(np.array(all_spike_times[cell_nb])>recording_start_time) & (np.array(all_spike_times[cell_nb])<recording_end_time)]
            data[cell_nb][rec_name]=(rec_spikes-recording_start_time)/fs    

            recording_start_time=onset
            rec_name=rec
    return data
    
     
        
        
#########################################
#####     Checkerboard Analysis     #####
#########################################



def get_recording_spikes(recording_name,all_recs_spikes):
    rec_spikes = {}
    for (cell_nb,recordings) in list(all_recs_spikes.items()):
        rec_spikes[cell_nb] = recordings[recording_name]
    return rec_spikes
    
def align_triggers_spikes(triggers, spike_times):
    # Clip the spike times to the recording time
    trigger_start = np.min(triggers)
    trigger_end = np.max(triggers)
    spike_times_filtered = spike_times[np.where((spike_times>trigger_start) & (spike_times<trigger_end))]

    # Set trigger & spikes start times to zero
    triggers = triggers - trigger_start
    spike_times_filtered = spike_times_filtered - trigger_start
    
    return triggers, spike_times_filtered

def build_rasters(cell_spikes, triggers, stim_frequency, nb_frames_by_sequence = params.nb_frames_by_sequence):

    nb_sequences      = int(len(triggers)/nb_frames_by_sequence)
    duration_sequence = int(nb_frames_by_sequence/stim_frequency)

    repeated_sequences_times = []
    spike_trains             = []
    spikes_counts            = np.zeros(int(nb_frames_by_sequence/2))

    analyse = {}
    for i in range(nb_sequences):
        
        # Get the repeated sequence times for the specified position
        time_start_id = i*nb_frames_by_sequence+int(nb_frames_by_sequence/2)
        time_end_id   = (i+1)*nb_frames_by_sequence
        times_sequence = triggers[time_start_id:time_end_id]
        repeated_sequences_times.append((times_sequence[0], times_sequence[-1]))

        # Build the spike trains corresponding to stimulus repetitions & make it start to 0
        spike_sequence = cell_spikes[np.where((cell_spikes>repeated_sequences_times[-1][0]) & (cell_spikes<repeated_sequences_times[-1][1]))]
        spike_trains.append(spike_sequence - repeated_sequences_times[-1][0])

        #Compute psth
        spikes_counts += np.histogram(spike_trains[-1], bins=int(nb_frames_by_sequence/2), range=(0,repeated_sequences_times[-1][1]-repeated_sequences_times[-1][0]))[0]
        
    analyse["spike_times"] = cell_spikes
    analyse["repeated_sequences_times"] = repeated_sequences_times
    analyse["spike_trains"] = spike_trains
    analyse["psth"] = spikes_counts / nb_sequences * stim_frequency #transform spikes_count in mean firing rates
    return analyse

def image_projection(image,mea=params.MEA):
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



def checkerboard_from_binary(nb_frames, nb_checks_x, nb_checks_y, checkerboard_file ,binary_source_path=params.binary_source_path, mea=params.MEA):
    
    binary_source_file = open(binary_source_path, mode='rb')
    checkerboard = np.zeros((nb_frames,nb_checks_x,nb_checks_y),dtype = 'uint8')
    
    for frame in tqdm(range(nb_frames)):
        
        image = np.zeros((nb_checks_x, nb_checks_y), dtype=np.float)
        
        for row in range(nb_checks_x):
            for col in range(nb_checks_y):
                bit_nb = (nb_checks_x*nb_checks_y * frame) + (nb_checks_x * row) + col
                binary_source_file.seek(bit_nb// 8)
                byte = int.from_bytes(binary_source_file.read(1), byteorder='big')
                bit = (byte & (1 << (bit_nb % 8))) >> (bit_nb % 8)
                if bit == 0:
                    image[row, col] = 0.0
                elif bit == 1:
                    image[row, col] = 1.0
                else:
                    message = "Unexpected bit value: {}".format(bit)
                    raise ValueError(message)        
        
        checkerboard[frame,:,:] = image_projection(image,mea)
    np.save(checkerboard_file, checkerboard)
    print(f'Checkerboard stimulus created and saved at : {checkerboard_file}')
    return checkerboard

def extract_from_sequence(cell_spikes, triggers, nb_repeats, stim_frequency, sequence_portion=(0.5,1), nb_frames_by_sequence = params.nb_frames_by_sequence):

    nb_sequences      = int(len(triggers)/nb_frames_by_sequence)
    duration_sequence = int(nb_frames_by_sequence/stim_frequency)

    repeated_sequences_times = []
    spike_trains             = []
    spikes_counts            = np.zeros((nb_sequences, int(nb_frames_by_sequence/2)))

    analyse = {}
    for i in range(nb_sequences):
        
        # Get the repeated sequence times for the specified position
        time_start_id = i*nb_frames_by_sequence+int(sequence_portion[0]*nb_frames_by_sequence)
        time_end_id   = i*nb_frames_by_sequence+int(sequence_portion[1]*nb_frames_by_sequence)
        times_sequence = triggers[time_start_id:time_end_id]
        repeated_sequences_times.append((times_sequence[0], times_sequence[-1]))

        # Build the spike trains corresponding to stimulus repetitions & make it start to 0
        spike_sequence = cell_spikes[np.where((cell_spikes>repeated_sequences_times[-1][0]) & (cell_spikes<repeated_sequences_times[-1][1]))]
        spike_trains.append(spike_sequence - repeated_sequences_times[-1][0])

        #Compute psth
        spikes_counts[i,:] = np.histogram(spike_trains[-1], bins=int(nb_frames_by_sequence/2), range=(0,repeated_sequences_times[-1][1]-repeated_sequences_times[-1][0]))[0]
        
    analyse["spike_times"] = cell_spikes
    analyse["repeated_sequences_times"] = repeated_sequences_times
    analyse["spike_trains"] = spike_trains
    analyse["counted_spikes"] = spikes_counts                                               
    analyse["psth"] = spikes_counts.sum(axis=0) / nb_repeats * stim_frequency #transform spikes_count in mean firing rates
    return analyse

def compute_3D_sta(data, checkerboard, stim_frequency, cluster_id=None, nb_frames_by_sequence=params.nb_frames_by_sequence, temporal_dimension = params.sta_temporal_dimension):
    
    nb_sequences = data["counted_spikes"].shape[0]
    sta = np.zeros_like(checkerboard[:temporal_dimension], dtype = 'float64')
    total_spikes = np.sum(data["counted_spikes"])
    
    for sequence in range(nb_sequences):
        for frame in range(temporal_dimension, int(nb_frames_by_sequence/2)):

                sta_frame_start  = sequence*int(nb_frames_by_sequence/2) + frame - temporal_dimension 
                sta_frame_end    = sequence*int(nb_frames_by_sequence/2) + frame                      
                weight = data["counted_spikes"][sequence,frame]

                sta += weight*checkerboard[sta_frame_start:sta_frame_end,:,:]
                
    if np.max(np.abs(sta)) > 0:        
        sta = sta/total_spikes
        #Bring values between -1 and 1
        sta -= np.mean(sta)
        sta /= np.max(np.abs(sta))
    else :
        print(f'Cluster {cluster_id} has no spikes, no sta can be found...')
    return sta


def gaussian2D(shape, amp, x0, y0, sigma_x, sigma_y, angle,):
    if sigma_x == 0:
        sigma_x = 0.001
    
    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]),int(shape[1]))
    x=np.linspace(0,shape[1],shape[1])
    y=np.linspace(0,shape[0],shape[0])
    X,Y = np.meshgrid(x,y)
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    
    return amp*np.exp( - (a*np.power((X-x0),2) + 2*b*np.multiply((X-x0),(Y-y0))+ c*np.power((Y-y0),2)))

def gaussian2D_flat(x, amp, x0, y0, rx, ry, rot):
    return gaussian2D(x, amp, x0, y0, rx, ry, rot).flatten()

def reduced_gaussian2D(x, amp, sigma_x, sigma_y, angle,):
    
    shape = (int(x[0]),int(x[1]))
    x0 = int(x[2])
    y0 = int(x[3])
    
    x=np.linspace(0,shape[1],shape[1])
    y=np.linspace(0,shape[0],shape[0])
    X,Y = np.meshgrid(x,y)
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    
    return amp*np.exp( - (a*np.power((X-x0),2) + 2*b*np.multiply((X-x0),(Y-y0))+ c*np.power((Y-y0),2)))

def reduced_gaussian2D_flat(x, amp, rx, ry, rot):
    return reduced_gaussian2D(x, amp, rx, ry, rot).flatten()

def gaussian_ellipse(amp, x0, y0, sigma_x, sigma_y, angle, ratio = math.sqrt(2)):

    level = amp*0.5
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    


    lim = math.sqrt((c*math.log(level/amp))/(b**2-c*a))
    xmin = x0 - lim
    xmax = x0 + lim
    
    X  = np.linspace(xmin,xmax,1000)
    Ym = y0 + (-2*b*(X-x0)-np.sqrt(4*b**2*(X-x0)**2 - 4*c*(a*(X-x0)**2+math.log(level/amp))))/2*c
    Yp = y0 + (-2*b*(X-x0)+np.sqrt(4*b**2*(X-x0)**2 - 4*c*(a*(X-x0)**2+math.log(level/amp))))/2*c

    return np.append(X,X), np.append(Ym,Yp)

####  Gabriel's analysis  ####

def gabriel_preprocessing(sta_3D, nb_frames = 15, kernel_lenght = 2, tresholding_factor = 2):
    
    data = sta_3D[-nb_frames:,:,:]
    
    #smoothing along time    
    kernel = np.ones(kernel_lenght)[:,None,None]/kernel_lenght
    data = convolve(data, kernel, mode='nearest')
    
    ## Take variance
    data = data.var(0)
    data -= np.median(data)
    data /= np.max(np.abs(data))
    spatial_sta = data.copy()
    
    ## Thresholding
    tresholding_factor = 2
    k_gauss = 1.5 # 1.5 mad ~ 1 std for gaussian noise
    mad = np.median(np.abs(data-np.median(data)))
    data[data<tresholding_factor*k_gauss*mad] = 0
    
    return data, spatial_sta

def gabriel_temporal_sta(sta_3D, gaussian_params):
    
    shape = (sta_3D.shape[1], sta_3D.shape[2])
    smoothing_kernel  = gaussian2D(shape, *gaussian_params)
    smoothing_kernel /= np.sum(smoothing_kernel)
    
    # Find max in space
    smoothed_sta = convolve(sta_3D.var(0), smoothing_kernel, mode='nearest')
    x_max, y_max = np.unravel_index(np.argmax(smoothed_sta), shape=shape)
    # Gaussian weighting kernel

    gaussian_kernel = gaussian2D(shape, gaussian_params[0], x_max, y_max, *gaussian_params[3:])
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
    # Weighted temporal trace
    return np.mean(gaussian_kernel[None,:,:]*sta_3D, (1,2))
    
def fit_gaussian(sta_spatial):
    
    center = np.unravel_index(np.argmax(sta_spatial, axis=None), sta_spatial.shape)
    guess  = [np.max(sta_spatial), center[1], center[0], 1, 1, 0]    
    
    xdata  = sta_spatial.shape
    ydata  = sta_spatial.flatten()
    
    ellispe_params_bounds = ((-2, 0, 0, 0.1, 0.1, 0),(2, sta_spatial.shape[0], sta_spatial.shape[0], sta_spatial.shape[0], sta_spatial.shape[0], 180))

    return curve_fit(gaussian2D_flat, xdata, ydata, p0=guess, bounds=ellispe_params_bounds)

def analyse_sta_gab(sta, cell_id):
    sta_3D = sta.copy()
    fitting_data, spatial_sta = gabriel_preprocessing(sta_3D)
    try:
        ellipse_params, cov = fit_gaussian(fitting_data)
    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay' : np.nan}
    temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
    
    return {'Spatial':spatial_sta, 'Temporal':temporal_sta, 'EllipseCoor':ellipse_params, 'Cell_delay' : np.nan}
    
####  Matias sta analysis ####

def smooth_sta(sta, alpha, max_time_window=15):
    pading_size = 1
    paded_sta       = np.pad(sta, pading_size)    ### change from a zeros matrice to a padded one. Countour of rf is not 0 now but the sta value itself
    receptive_field = np.zeros(sta.shape)
    for x in range(sta.shape[1]):
        for y in range(sta.shape[2]):
            receptive_field[:,x,y] = paded_sta[1:-1,x+pading_size,y+pading_size]*alpha + (1-alpha)*paded_sta[1:-1,x+pading_size-1:x+pading_size+2,y+pading_size-1:y+pading_size+2].sum(axis=(1,2)) 
    
    best    = np.unravel_index(np.argmax(np.abs(receptive_field[-max_time_window:,:,:])), receptive_field.shape)
    best_t = best[0] + max(sta.shape[0]-max_time_window,0)
    
    return receptive_field, (best_t,best[1],best[2]), receptive_field[best]


def get_cell_shift(sta):
    sta_3D = sta.copy()
    smooth_sta_1_3D, best_1,max_val1 = smooth_sta(sta_3D, alpha=0.5)
    smooth_sta_2_3D, best_2 ,max_val2 = smooth_sta(sta_3D, alpha=0.8)
    
    if abs(max_val1) > abs(max_val2):
        return best_1
    else:
        return best_2

def preprocess_fitting_matias(spatial, treshold=0.1):
    
    sta_spa = spatial.copy()
    sta_treshold = np.max(np.abs(spatial))*treshold
    sta_spa[np.abs(sta_spa)<sta_treshold] = 0
    return sta_spa

def matias_temporal_spatial_sta(sta_3D):
    if np.max(np.abs(sta_3D)) == 0:
        print(f'Cell {cell_id} : Could not find sta')
        return 'Error detected : 3D sta empty', 'Error detected : 3D sta empty'
    
    (best_t, best_x, best_y)  = get_cell_shift(sta_3D)
    sta_temporal  = sta_3D[:,best_x,best_y]
    sta_spatial   = sta_3D[best_t,:,:]
    sta_spatial  /= np.max(np.abs(sta_spatial)) 
    
    return sta_temporal, sta_spatial, (best_t, best_x, best_y)


def double_gaussian_fit(spatial):
    
    center = np.unravel_index(np.argmax(np.abs(spatial), axis=None), spatial.shape)
    ydata = spatial.flatten()
    
    #First fit without center variability
    first_guess = [spatial[center[0],center[1]], 1, 1, 0]   
    xdata = [spatial.shape[0],spatial.shape[1],center[1],center[0]]
    
    ellispe_params_bounds = ((-2, 0.1, 0.1, 0),(2, spatial.shape[0], spatial.shape[0], 180))

    opt, cov = curve_fit(reduced_gaussian2D_flat, xdata, ydata, p0=first_guess, bounds=ellispe_params_bounds)

    #Second fit with center variability
    xdata = spatial.shape
    second_guess = [opt[0], center[1], center[0], opt[1], opt[2], opt[3]]

    ellispe_params_bounds = ((-2, 0, 0, 0.1, 0.1, 0),(2, spatial.shape[0], spatial.shape[0], spatial.shape[0], spatial.shape[0], 180))
    return curve_fit(gaussian2D_flat, xdata, ydata, p0=second_guess, bounds=ellispe_params_bounds)
    
def analyse_sta_matias(sta, cell_id):
    sta_3D = sta.copy()
    
    sta_temporal, sta_spatial, best = matias_temporal_spatial_sta(sta_3D)
    fitting_data = preprocess_fitting_matias(sta_spatial)
    try :
        ellipse_params,cov = double_gaussian_fit(fitting_data)
    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay' : best[0]}
    return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':ellipse_params, 'Cell_delay' : best[0]}

### Guilhem sta analysis ### (mixed between both)

def analyse_sta(sta, cell_id):
    sta_3D = sta.copy()
    fitting_data, spatial_sta = gabriel_preprocessing(sta_3D, tresholding_factor=1)
    try:
        ellipse_params, cov = double_gaussian_fit(fitting_data)
        temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
        best_t  = np.argmax(np.abs(temporal_sta[-15:]))
        best_t += max(sta.shape[0]-15,0)

        spatial_sta = sta_3D[best_t]

        return {'Spatial':spatial_sta, 'Temporal':temporal_sta, 'EllipseCoor':ellipse_params, 'Cell_delay' : best_t}

    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':spatial_sta, 'Temporal':np.zeros(40), 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay':np.nan}
    

### Tom sta analysis ### (new fitting of ellipse with new denoising and smoothing of STAs)

def preprocess_fitting_tom(sta):

    shape0,shape1=sta.shape
    denoised_sta=np.zeros([shape0,shape1])
    enlarged_sta=np.zeros([shape0+2,shape1+2])

    enlarged_sta[1:shape0+1,1:shape0+1]=sta

    for x in range(shape0):
        for y in range(shape1):
            denoised_sta[x,y]=(np.sum(enlarged_sta[x:x+2,y:y+2]*0.2)+enlarged_sta[x+1,y+1]*0.8)/2.4

    sta=denoised_sta
    
    expon_treat= 1.25
    vmax_thresh = 2 
    to0 = 0.2001
    cmap='RdBu_r'
    put_to0 = np.exp(np.log(to0)*expon_treat) 
    
    sta=np.sign(sta)*np.exp(np.log(abs(sta))*expon_treat)
    vmax= np.max([np.amax(sta),-np.amin(sta)])  *vmax_thresh
    sta[abs(sta)<vmax*put_to0]=0

    return sta


def analyse_sta_tom(sta, cell_id):
    sta_3D = sta.copy()
    
    sta_temporal, sta_spatial, best = matias_temporal_spatial_sta(sta_3D)
    fitting_data = preprocess_fitting_tom(sta_spatial)
    try :
        ellipse_params,cov = double_gaussian_fit(fitting_data)
    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay' : best[0]}
    return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':ellipse_params, 'Cell_delay' : best[0]}
    


def plot_sta(ax, spatial_sta, ellipse_params, level_factor=0.4):
    #magnified_ellipse_params=(np.array(ellipse_params)*[gaussian_factor, 1,1,gaussian_factor,gaussian_factor,1])
    gaussian = gaussian2D(spatial_sta.shape,*ellipse_params)
    ax.imshow(spatial_sta)
    if ellipse_params[0] != 0:
        ax.contour(np.abs(gaussian),levels = [level_factor*np.max(np.abs(gaussian))], colors='w',linestyles = 'solid', alpha = 0.8)
    return ax

#New display with max and min equal and new coulor
def plot_sta_tom(ax, spatial_sta, ellipse_params, level_factor=0.4):

    gaussian = gaussian2D(spatial_sta.shape,*ellipse_params)

    vmax=np.max([np.amax(spatial_sta),-np.amin(spatial_sta)])
    ax.imshow(spatial_sta,cmap='RdBu_r',vmax=vmax,vmin=-vmax)
    if ellipse_params[0] != 0:
        ax.contour(np.abs(gaussian),levels = [level_factor*np.max(np.abs(gaussian))], colors='y',linestyles = 'solid', alpha = 0.4,lw=5)
    return ax


### Analysis to quantify the presence of STAs

def SNR_test(sta,contour): #Calculate the SNR of cells
    
    path = mpltPath.Path(contour[0][0])
    points=[]
    
    for x_id in range(sta.shape[0]):
        for y_id in range(sta.shape[1]):
                points.append([x_id,y_id])
    
    inside = path.contains_points(points)

    noise=[]
    signal=[]
    for ins_id in range(len(inside)):
        if inside[ins_id]==False:
            noise.append(sta[points[ins_id][1],points[ins_id][0]])
        else:
            signal.append(sta[points[ins_id][1],points[ins_id][0]])
            noise.append(0)

    nb_in=len(signal)
    nb_out=len(noise)

    noise_compression=[]

    for nb_comp in range(sta.shape[0]):
        noise_compression.append(np.mean(noise[nb_comp*40:(nb_comp+1)*40]))
        
    noise=np.sum(np.abs(noise_compression))
    signal=np.abs(np.sum(signal))

    SNR=signal/noise

    return SNR


def PolyArea(x,y): #Used to calculate an area of a polygon
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def check_presence_STA(sta,ellipse_coor,nb_of_pixels_by_check,tresh_snr=2.75,level_factor=0.2): #Used to check the presence of STAs

    pxl_size_dmd=params.pxl_size_dmd
    
    gaussian = gaussian2D(sta.shape,*ellipse_coor)

    x0=ellipse_coor[1]
    y0=ellipse_coor[2]

    #See if the STA is in the center 
    xshape=sta.shape[0]
    yshape=sta.shape[1]
    if x0>0.8*xshape or x0<0.2*xshape or y0>0.8*yshape or y0<0.2*yshape:
        return [0.1,0.1]

    #See if the STA has a fitted ellipse
    if ellipse_coor[0] != 0:
        fig=plt.figure()
        cs=plt.contour(np.abs(gaussian),levels = [level_factor*np.max(np.abs(gaussian))])
        contour=cs.allsegs
        plt.close()

        #Verify that the diameter of the ellipse is neither too big nor too small
        area=PolyArea(contour[0][0][:,0],contour[0][0][:,1])
        diameter=2*np.sqrt(area/np.pi)*nb_of_pixels_by_check*pxl_size_dmd

        if diameter<100 or diameter>500:
            return [0.3,diameter]

        #Verify that the SNR is superior to the threshold of the SNR
        if SNR_test(sta,contour)<tresh_snr:
            return [0.4,SNR_test(sta,contour)]
            
    else:
        return [0.2,0.2]

    return [1,SNR_test(sta,contour)]


#############################################
######         Drifting Gratings       ######
#############################################




def compute_tuning(ch_raster,base_fire, seq_len, seq_sep, n_repeats=4):
    ###########################################################
    # computing tuning
    merged = list(itertools.chain(*ch_raster))   #all the spike times of all the 32 gratings. In this way when I bin I am
                                                 #binning per each of the 8 angles the responses to all the 4 repetitions of
                                                 #that angle

    nbins = 8*10*20                 #totoal nb of bins  (1600)
    binsize = seq_sep*8*1000//nbins #bin size in ms     (100)
    binsec = 1000//binsize          #nb bins per second  (10)
    base_fire=base_fire*(seq_sep*8/nbins)*n_repeats
    
    bins =  np.linspace(0,seq_sep*8,nbins+1)
    counts, bins = np.histogram(merged,bins=bins)   #binning the spike times of all the repetitions at once
    counts=counts-base_fire
    maxcount = np.amax(counts)

    #for plotting purposes, counts has 1600 bins, 10 each second of the 160 seconds. But some of this bins are fake because
    #the seq_sep (20 secs for the slow gratings) added in ch_raster is longer than the actual seq_len (12 secs for slow grating),
    #in which the stimulus was presented. So the last 8 secs after each angle have to have 80 empty.
    
        #--------------------------
    TuneSum = np.zeros(9)
    VxS=0
    VyS=0

    for a in np.arange(8):
        #################################################
        #per each angle I select the bins that go from 2 secs after the grating onset to the grating offset. Why?
        sel_bins = np.copy(counts[ int(seq_len*1000/6)//binsize + int(seq_sep*binsec*a): int(seq_len*binsec + seq_sep*binsec*a)])  
        #################################################

        TuneSum[a] = np.sum(sel_bins)    #per each angle these are all the spikes that the cell fired during the 4 repetitions
                                         #of that angle from 2 to 12 seconds
        #print(TuneSum[a])
        VxS+= np.cos(np.pi*a*45/180)*TuneSum[a]
        VyS+= np.sin(np.pi*a*45/180)*TuneSum[a]
#             VxM+= np.cos(np.pi*a/180)*TuneMax[a]
#             VyM+= np.sin(np.pi*a/180)*TuneMax[a]
        if a==0:
            TuneSum[a+8] = np.sum(sel_bins)

############################    
    if sum(TuneSum)==0:
        DG_data = ({'IDX':0,'Tuning':TuneSum,'atune':0,'Rtune':0, 'rasters': np.zeros( (4,len(bins)) ),'counts':counts, 'maxcount':maxcount, 'bins':bins})
        return np.zeros(9), 0, 0, 0, counts, maxcount, bins, DG_data
############################
    VxS=VxS/np.amax(TuneSum) 
    VyS=VyS/np.amax(TuneSum) 

    TuneSum=TuneSum/np.amax(TuneSum)
    atune = np.arctan2(VyS,VxS)
    R = np.sqrt(VyS**2+VxS**2)
   
    angle = int(np.round(atune/np.pi*4 ))
        
    IDX = (TuneSum[:-1][angle]-TuneSum[:-1][int((angle+4)%8)])/(TuneSum[:-1][angle]+ TuneSum[:-1][int((angle+4)%8)])
    if IDX<-0.2:
        angle2=angle+1
        IDX = (TuneSum[:-1][angle2]-TuneSum[:-1][int((angle2+4)%8)])/(TuneSum[:-1][angle2]+ TuneSum[:-1][int((angle2+4)%8)])
        angle=angle2
    if IDX<-0.2:
        angle2=angle-2
        IDX = (TuneSum[:-1][angle2]-TuneSum[:-1][int((angle2+4)%8)])/(TuneSum[:-1][angle2]+ TuneSum[:-1][int((angle2+4)%8)])
        if IDX<-0.2: angle=angle+1
        IDX = (TuneSum[:-1][angle]-TuneSum[:-1][int((angle+4)%8)])/(TuneSum[:-1][angle]+ TuneSum[:-1][int((angle+4)%8)])
    
    DG_data = ({'IDX':IDX,'Tuning':TuneSum,'atune':atune,'Rtune':R, 'rasters': ch_raster, 'counts':counts, 'maxcount':maxcount, 'bins':bins})

    ###########################################################
    return TuneSum, atune, R, IDX, counts, maxcount, bins, DG_data




#############################################
######            Clustering           ######
#############################################

def cell_selection_for_clustering(cells, CT_directory_path, selected_cells_sta=[], selected_cells_chirp=[]):
    print("Selecting via STA ...")
    if selected_cells_sta == []:
        for cell_nb in tqdm(cells):
            plt.figure(r"Current cell", figsize=(10,10))
            image = np.asarray(plt.imread(os.path.normpath(os.path.join(CT_directory_path,r'{}_Chirp_raster+STA.png'.format(cell_nb)))))
            plt.imshow(image[50:220,1330:1550])
            plt.axis('off')
            plt.show(block=False)
            time.sleep(0.2)
            if input("Keep cell {} for clustering using sta? Type Yes to select as good : ".format(cell_nb)) in ["Y", "Yes", "y", "yes"]:
                selected_cells_sta += [cell_nb]
                
    print("List of selected cells using sta : ", selected_cells_sta)
    print("Selecting via chirp ...")

    if selected_cells_chirp == []:
        for cell_nb in tqdm(cells):
            plt.figure(r"Current cell", figsize=(50,100))
            image = np.asarray(plt.imread(os.path.normpath(os.path.join(CT_directory_path,r'{}_Chirp_raster+STA.png'.format(cell_nb)))))
            plt.imshow(image[:,:1350])
            plt.show(block=False)
            time.sleep(0.2)
            if input("Keep cell {} for clustering using chirp? Type Yes to select as good : ".format(cell_nb)) in ["Y", "Yes", "y", "yes"]:
                selected_cells_chirp += [cell_nb]
    print("List of selected cells using chirp : ", selected_cells_chirp)
    
    selected_cells = [id for id in selected_cells_sta if id in selected_cells_chirp]
    
    return selected_cells, selected_cells_sta, selected_cells_chirp


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
def restrict_array(array, value_min, value_max):
    array = array[array>=value_min]
    array = array[array<=value_max]
    return array.tolist()


def correlate_PersonPM(cell1,cell2, max_shift=25):
    assert max_shift<max(len(cell1),len(cell2))
    center = np.corrcoef(cell1,cell2)[0,1]
    right  = []
    left   = []
    for t in range(1,max_shift+1):
        right.append(np.corrcoef(cell1[t:],cell2[:-t])[0,1]) 
        left.append(np.corrcoef(cell1[:-max_shift+t-1],cell2[max_shift-t+1:])[0,1])
    return np.asarray(left + [center] + right)


def noise_and_stim_correlations(resp_cell1, resp_cell2, max_shift=25, shift_time_resolution=1):
    """
        Exactly the same as above but manually computed. Not in use.
    """
    #resp_cell should be of the form (nb_trials, nb_response points)
    #THIS MIGHT HAVE NORMALIZATION PROBLEMS IN CASE OF CURRENTS!!!!
    noise_corr=[]
    stim_corr=[]
   
    for lag in range(-max_shift,max_shift+1,shift_time_resolution):
        shifted_c2=np.roll(resp_cell2, lag, axis=0)
       
        V_1=((resp_cell1-resp_cell1.mean())**2).mean()
        V_2=((shifted_c2-shifted_c2.mean())**2).mean()
       
        nc=( (resp_cell1 -  resp_cell1.mean(axis=0)) * (shifted_c2- shifted_c2.mean(axis=0)) ).mean()-np.sqrt(V_1*V_2)
        noise_corr.append(nc)
#         tot_corr=((resp_cell1-resp_cell1.mean())*(shifted_c2-shifted_c2.mean()) ).mean()/np.sqrt(V_1*V_2)
#         sc=tot_corr-nc
#         stim_corr.append(sc)
#     return np.array(noise_corr), np.array(stim_corr)
    return np.array(noise_corr)



#############################################
######          ID card                ######
#############################################

def find_Analysis_Directory(dir_type="Checkerboard", output_directory = params.output_directory):
    """
        Automatically calls for the analysis folder using names defined in the pipeline :
            - Checkerboard_Analysis_rec_i
            - DG_Analysis_rec_i
            - CellTyping_Analysis_rec_i
            
        dir_type should be either "Checkerboard", "DG", or "CellTyping"
        
        If severeal analysis has been done for the same type, you will have to input the one to select.
    """
    assert dir_type in ["Checkerboard", "DG", "CellTyping"]
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


def get_cell_rpvs(cells, phy_directory, rpv_len=2.0, fs=20000):
    
    spike_clusters = np.load(os.path.join(phy_directory, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(phy_directory, 'spike_times.npy'))
    
    cell_rpv ={}

    for cell_nb in cells:

        cell_rpv[cell_nb] = {}
        
        sp_times=spike_times[np.where(spike_clusters==cell_nb)[0]]/fs  #cell's sp_times in seconds
        
        interspike_intervals = compute_interspike_intervals(sp_times)  #np.diff
        
        #percentage of isi less than rpv_len milliseconds
        rpv = compute_refractory_period_violation(sp_times,duration=rpv_len, cell_nb=cell_nb) 
        
        nb_rpv = compute_number_of_rpv_spikes(sp_times,duration=rpv_len)

        cell_rpv[cell_nb]["nb_spikes"] = len(sp_times)
        cell_rpv[cell_nb]["isi"] = interspike_intervals
        cell_rpv[cell_nb]["rpv"] = rpv
        cell_rpv[cell_nb]["nb_rpv_spikes"] = nb_rpv

    return cell_rpv


def compute_interspike_intervals(spike_times):

    return np.diff(spike_times).astype(np.float64)

def compute_number_of_rpv_spikes(spike_times, duration=2.0):
    
    isis = compute_interspike_intervals(spike_times)
    nb_rpv = np.count_nonzero(isis <= 1e-3 * duration)
    
    return float(nb_rpv)

def compute_refractory_period_violation(spike_times, duration=2.0, cell_nb=None):
    """
    spike_times : the spike times of the neuron to study
    duration : the duraiton of the refractory period, in ms
    """

    isis = compute_interspike_intervals(spike_times)
    nb_isis = len(isis)
    if nb_isis==0: 
        if cell_nb==None: print('This cell has no spikes')
        else: print('Cell {} has no spikes'.format(cell_nb))
        return 0
    else:
        rpv = compute_number_of_rpv_spikes(spike_times, duration)/ float(nb_isis) *100
        return rpv

    
###########################################################
###########          Analysis from vec          ###########
###########################################################
    
    
    
def split_spikes_between_triggers(spike_train,triggers):
    """
Returns a list of spikes includes between 2 triggers in a row. Everything must be in sampling point or sec.
    """
    return [spike_train[(spike_train >= triggers[i]) & (spike_train < triggers[i + 1])] for i in range(len(triggers) - 1)]

def get_sequences_triggers(triggers, vec):
    """
Spilt all triggers into dict of triggers from the same sequence using the keys provided in vec.
Same key for the triggers means same sequence.

Could be rewritten without the "defaultdict" trick
    """
    from collections import defaultdict
    sequences = defaultdict(list)

    keys = vec.astype(int).astype(str)
    for key, trigger in zip(keys, triggers):
        sequences[key].append(trigger)
    
    return dict(sequences) #this dictionnary has its keys ordered as the vec. !! CAUTION !! works for python > 3.7 only
   
    
def get_spikes_sequences(spike_times, trig_seq):
    """
Read the first trigger of all sequence and group all spikes between each begining of sequence into a dict with
sequence key as dict key and a list of spike times with the 0 at the begining of a sequence.
    """
    trigs=[[trig_list[0],trig_list[-1]+np.mean(np.diff(np.array(trig_list)))] for trig_list in trig_seq.values()]  #make a list of all first and last trig of each seq    
    splited_spikes = [split_spikes_between_triggers(spike_times,seq_times)[0] for seq_times in trigs]
    return dict(zip(trig_seq.keys(), splited_spikes))
    
    
def spikeseq2raster(spikesequences, trig_seq):
    """
Makes a raster from a dictionnary of sequences splited with repetition. 
Looks for the key to stack repetitions (last 2 digits of the key). Repetition number is not representative of when it has been played 
    """
    
    from collections import defaultdict
    rasters = defaultdict(list) #more compliant than dict. Allows you to either use an existing key or create it with empty list and than use it if missing.

    for key in spikesequences.keys():
        rasters[key[:-2]].append(spikesequences[key]-trig_seq[key][0])
    return dict(rasters)

def spikeseq2psth(raster, trig_seq, n_bin=40):
    psth={}
    for key in raster.keys():
        n_rep = len(raster[key])
        if key=='':
            seq_range  = (0, trig_seq['0'][-1]-trig_seq['0'][0] + np.mean(np.diff(trig_seq['0'])))
        else:
            seq_range  = (0, trig_seq[key+'00'][-1]-trig_seq[key+'00'][0] + np.mean(np.diff(trig_seq[key+'00'])))
        
        if n_bin =="relative":
            all_spikes_times=[]
            for i in range(n_rep):
                all_spikes_times+=list(raster[key][i])
            psth[key] = np.histogram(np.array(all_spikes_times), bins=max(1,int(np.sqrt(len(all_spikes_times)))), range=seq_range   )[0]/n_rep
        else:
            binned_spike_count = np.zeros((n_rep, n_bin))
            for i in range(n_rep):
                binned_spike_count[i,:] = np.histogram(raster[key][i], bins=n_bin, range=seq_range   )[0]
            psth[key] = np.sum(binned_spike_count, axis=0)/n_rep
            
    return psth

def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    """
Function to smooth a 1D numpy array before plotting
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def reshape_dict(original_dict):
    """
This function allows you to reshape dictionnaries by reversing their keys. 
If you have {Cell1 : {key1: data, key2: data}, Cell2 : {key1: data, key2: data}}
you will get {key1 : {Cell1: data, Cell2: data}, key2 : {Cell1: data, Cell2: data}}
    """
    reshaped_dict = {}

    for cell_number, seq_dict in original_dict.items():
        for seq_number, data_dict in seq_dict.items():
            if seq_number not in reshaped_dict:
                reshaped_dict[seq_number] = {}
            reshaped_dict[seq_number][cell_number] = data_dict

    return reshaped_dict

###########################################################
###########          Registration Holo          ###########
###########################################################

# Functions
def buildH(t_pre,s,t_post,r=0):
    H_pre_translation = np.array([[1, 0, t_pre[1]],
                                  [0, 1, t_pre[0]],
                                  [0, 0,    1]])
  
                          
    H_rotation = np.array([[cos(r),  -sin(r), 0],
                          [sin(r),   cos(r), 0],
                          [0,        0,      1]])
    
    H_scaling = np.array([[s[0], 0,   0],
                          [0,   s[1], 0],
                          [0,    0,   1]])
    
    H_post_translation = np.array([[1, 0, t_post[1]],
                                   [0, 1, t_post[0]],
                                   [0, 0, 1]])
    return H_post_translation@H_rotation@H_scaling@H_pre_translation


def transform_coordinates(coordinates, homography):
    coordinates = np.append(coordinates, np.array([1]))
    transformation = homography@coordinates.T
    transformation = transformation / transformation[2]
    transformed_coordinates = np.array(transformation[:2])
    return transformed_coordinates


def get_ellipse(parameters,factor=2):
        
    amplitude, x0, y0, sigma_x, sigma_y, theta = parameters
    width = factor * 2.0 * sigma_x
    height = factor * 2.0 * sigma_y

    t = np.linspace(0, 2*np.pi, 360)
    
    Ell = np.array([sigma_x*np.cos(t) , sigma_y*np.sin(t)])
    
    R_rot = np.array([[np.cos(-np.deg2rad(theta)) , -np.sin(-np.deg2rad(theta))]
                      ,[np.sin(-np.deg2rad(theta)) , np.cos(-np.deg2rad(theta))]])  
    
    Ell = np.dot(R_rot, Ell)
    Ell[0,:] += x0
    Ell[1,:] += y0
    ell_size = np.abs(np.pi*width*height)
    ell_meas = 1-min(width, height)/max(width, height)
    
    return Ell, ell_size, ell_meas

def find_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def find_aligned_point(point, ellipse, sanity_check=True):
    ellipse_center = np.mean(ellipse, axis=1)
    index = 0

    angle = 10000
    
    for i in range(360):
        angle_temp = find_angle(point,ellipse_center, ellipse[:,i])
        if angle_temp < angle:
            angle = angle_temp
            index = i
            
    closest_point = ellipse[:,index]
            
    if sanity_check:
            
        plt.figure()
        plt.plot(ellipse[1],ellipse[0])
        plt.scatter(ellipse_center[1],ellipse_center[0], marker='+')
        plt.scatter(closest_point[1],closest_point[0], color="green")
        plt.scatter(point[1],point[0], color='r')
        
            
    return closest_point


def compute_distance_between_points(point_1, point_2):
    distance = np.linalg.norm(point_1 - point_2)
    return distance
