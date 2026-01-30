"""Configure your experimental pipeline.

contact: laquitainesteeves@gmail.com

This is an example template for how to configure your experimental
pipeline.

Use Ctrl+F to look for a specific parameter.

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""
import os

def setup_threshold_pxl_size_size_dmd(params: dict):
    """setup the optimal threshold for detecting stimuli,
    the size of one pixel of the DMD in µm?
    on the camera or in reality? ("pixel size DMD") and
    the dimension of the DMD ("size dmd")
    note: the threshhold onsets varies with the rig
    """
    if params["MEA"] == 1:
        threshold = 270e3
        size_dmd = None
        pxl_size_dmd = None
    elif params["MEA"] == 2:
        threshold = 150e3
        size_dmd = [864, 864]  # dimensions of the DMD, in pixels
        pxl_size_dmd = (
            3.5  # The size of one pixel of the DMD in µm? on the camera or in reality?
        )
    elif params["MEA"] == 3:
        threshold = 170e3
        size_dmd = [760, 1020]
        pxl_size_dmd = 2.5
    elif params["MEA"] == 4:
        threshold = -3.14470e5
        size_dmd = None
        pxl_size_dmd = None
    else:
        raise ValueError("MEA is not defined in params")
    return threshold, pxl_size_dmd, size_dmd


def make_dict_keys_global_variables(params: dict):
    for k, v in params.items():
        globals()[k] = v

# setup experiment parameters (always check!)
basic_params = {
    "root": r"./data/20251219_PulsingGratings_PupilSize",  # This is the root folder of your experiment; all other files must be inside of this folder or manually specified.
    "exp": r"20251219_PulsingGratings_PupilSize",  # name of your experiment for saving the triggers
    "MEA": 3,  # select MEA (3=2p room) (4=MEA1 Polychrome)
    "raw_files_folder": r"RAW_Files",  # Enter the name of the folder containing all your raw files. It will be conctenated with root to find your raws. If the folder is not in root, change the variable "recording_directory" manually.
    "recording_names": [
        "00_AccCheck_30Hz_16px_42sq_50%30ND",
        "01_Swn_30Hz_48pxCh_6pxL_50%30ND",
        "02_Chirp_50Hz_50%30ND",
        "03_DG_50Hz_50%30ND", 
        "04_PulsingGratings-PS0_40Hz_50%30ND",
        "05_PulsingGratings-PS1_40Hz_50%30ND",
        "06_PulsingGratings-PS2_40Hz_50%30ND",
    ],  # Ordered list of recording_names without your file extension (mostlikly .raw). Don't forget to put it as raw string using r before the name : r'Checkerboard'.
    "registration_directory": r"",
}

# setup MEA parameters (always check!)
mea_params = {
    "mea_spacing": 30,  # the spacing between two electrodes of the MEA in µm for registration
    "n_electrodes": 16,  # number of electrodes on one side of the MEA. N tot electrodes = n_electrodes**2
}


########################################################
# Path Utils
########################################################

def find_files(path: str):
    """
    Function to get all recording files name from either a txt file name or a folder.
    Detects raw files automatically

    Input :
        - path (string) : a .txt file path containing the recording .raw files name
        - path (string) : a folder path containing all the recordings .raw files in alphabetic order

    Output :
        - (list) a list of strings of files names matching the recordings names

    Possible mistakes :
        - File names are written in .txt without the '.raw' extension
        - Several files on the same line
        - Wrong file/folder path
        - Other files not in '.raw' extension in the folder
        - Files names aren't ordered
    """
    # Check if given path is a file and if it exist
    if os.path.isfile(os.path.normpath(path)):

        # If yes, than open in variable "file"
        with open(os.path.normpath(path)) as file:

            # return the text of each line as a file name ordered from top to bottom
            return file.read().splitlines()

    # If no, the path is considered as a folder and return the name of all the files in alphabetic order
    return sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(path)
            if (
                os.path.isfile(os.path.join(path, f))
                and os.path.splitext(f)[1] == ".raw"
            )
        ]
    )


def create_path_automatically(params: dict):

    print("\n-------- Creating all paths ---------\n")

    # Link to the actual raw files frome the recording
    # listed in the input_file
    recording_directory = os.path.join(params["root"], params["raw_files_folder"])

    # Link to the folder where spiking circus will look
    # for the symbolic links "recording_0i.raw"
    symbolic_link_directory = os.path.join(params["root"], r"Sorting")
    if not os.path.exists(symbolic_link_directory):
        os.makedirs(symbolic_link_directory)
        print(f'- Created "Sorting" path: {symbolic_link_directory}')
    else:
        print(f'- "Sorting" path already exists')

    # copy path
    sorting_directory = symbolic_link_directory

    # link to .GUI directory where phy extracts all
    # arrays and data on spikes (folder name ends by .GUI)
    phy_directory = os.path.normpath(
        os.path.join(symbolic_link_directory, r"recording_00/recording_00.GUI")
    )

    # Link to the directory where output data should be saved
    output_directory = os.path.join(params["root"], r"Analysis")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f'- Created "output" path: {output_directory}')
    else:
        print(f'- "output" path already exists')

    # Link to the folder in which triggers will be saved.
    # If doesn't exist, will be created.
    triggers_directory = os.path.join(output_directory, "triggers")
    if not os.path.exists(triggers_directory):
        os.makedirs(triggers_directory)
        print(f'- Created "triggers" path: {triggers_directory}')
    else:
        print(f'- "triggers" path already exists')

    # Path to the checkerboard binary file used to generate stimuli
    binary_source_path = "./ressources/binarysource1000Mbits"

    raw_filtered_directory = os.path.join(params["root"], "RAW_filtered")

    registration_frames = os.path.join(
        params["root"], params["registration_directory"] + r"/frames/"
    )

    registration_imgs = os.path.join(
        params["root"], params["registration_directory"] + r"/imgs/"
    )

    # Do not use this unless you know how !!!
    recording_names = find_files(recording_directory)
    return (
        recording_directory,
        symbolic_link_directory,
        sorting_directory,
        phy_directory,
        output_directory,
        triggers_directory,
        binary_source_path,
        raw_filtered_directory,
        registration_frames,
        registration_imgs,
        recording_names,
    )
    
# create paths automatically (change only if your file organization is specific!)
( recording_directory,            # Link to the actual raw files from the recording listed in the input_file
    symbolic_link_directory,
    sorting_directory,
    phy_directory,
    output_directory,               # Directory where preprocessing info are saved
    triggers_directory,             # folder in which the triggers are saved
    binary_source_path,
    raw_filtered_directory,
    registration_frames,
    registration_imgs,
    recording_names,                # Recordings labels available
) = create_path_automatically(basic_params)

# setup advanced parameters
# Default values used in utils functions. If a function has a wrong behaviour, you may want to look in here.
advanced_params = {
    "dtype": "uint16",  # Datatype used to open rawfiles recordings
    "voltage_resolution": 0.1042,  # µV / DC level, Resolution of one step of mea signal amplitude in micro volts
    "nb_bytes_by_datapoint": 2,  # Size of a sample in bytes
    "time": 10,  # Time in s at the begining of the recording used to check recording type
    "maximal_jitter": 0.25e-3,  # Maximal error admissible in sec for time gap between triggers
    "nb_frames_by_sequence": 1200,  # Number of frames in each checkerboard sequence
    "sta_temporal_dimension": 40,  # number of frames to look in for the lag
    "sta_smooth_value": 0.8,
    "sta_treshold": 0.1,
    "temporal_dimension": 30,
}

# Setup most advanced parameters (Only if you know what you are doing!).
# Those parameters are following the setups specs of january 2023
most_advanced_params = {
    "threshold": setup_threshold_pxl_size_size_dmd(basic_params)[0],
    "pxl_size_dmd": setup_threshold_pxl_size_size_dmd(basic_params)[1],
    "size_dmd": setup_threshold_pxl_size_size_dmd(basic_params)[2],
    "nb_channels": 256,  # 256 for standard MEA, 17 for MEA1 Polychrome
    "holo_channel_id": 127,  # MEA channel id containing holographic triggers trace
    "visual_channel_id": 126,
    "fs": 20000,  # number of triggers samples acquired per second (Sampling frequency of the MEA)
    "time_after": 10,  # Time (ms) before a trigger to remove from the spyking circus analysis due to photo induced current on mea
    "time_before": 10,  # Time (ms) after a trigger to remove  from the spyking circus analysis due to photo induced current on mea
    "offset_time": 0.5,  # Delay (sec) after a trigger to add a fake trigger in the data adding one more dead period
}

# setup pipeline parameters
# relative path from pipeline notebook to a folder containing ressources such as mea pictures and datasets
ressources = r"./ressources"

# make all dictionary keys global variable
# note: this is a very poor practice as we have 
# little control over the variables and packages
# available at any given time in our environment 
# This is done to minimize disruption for now.
# TODO: refactor utils.py to take in an dictionary
# of parameters as input
make_dict_keys_global_variables(basic_params)
make_dict_keys_global_variables(mea_params)
make_dict_keys_global_variables(advanced_params)
make_dict_keys_global_variables(most_advanced_params)

