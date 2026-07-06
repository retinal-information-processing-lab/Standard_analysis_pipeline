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
import glob

# setup pipeline parameters
# relative path from pipeline notebook to a folder containing ressources such as mea pictures and datasets
ressources = r"./ressources"

# setup experiment parameters (always check!)

basic_params = {
    "root": r"/media/idv-s8/SSD Storage/20260702_Brid_Vs_Retina_Vs_Strychnine_1_test_pipeline",  # This is the root folder of your experiment; all other files must be inside of this folder or manually specified.
    "exp": r"20260702_Brid_Vs_Retina_Vs_Strychnine_1",  # name of your experiment for saving the triggers
    "MEA": 2,  # select MEA (3=2p room) (4=MEA1 Polychrome)
    "raw_files_folder": r"RAW_Files",  # Enter the name of the folder containing all your raw files. It will be conctenated with root to find your raws. If the folder is not in root, change the variable "recording_directory" manually.
    "recording_names": [
        "20260702_00_SWN_30Hz",
        "20260702_01_SWN_30Hz",
        "20260702_02_chirp_50Hz",
        "20260702_03_DG_2sT_10rep_8dir_50Hz",
        "20260702_04_barcode_3dir_50Hz",
        "20260702_05_black_eagle_CTL_40Hz",
        "20260702_05_black_eagle_CTL_40Hz_part2",
        "20260702_06_SWN_drug_being_added_30Hz",
        "20260702_07_black_eagle_strychnine_40Hz",
        "20260702_08_SWN_drug_being_removed_30Hz",
        "20260702_09_black_eagle_post_strychnine_CTL_40Hz",
    ],  # Ordered list of recording_names without your file extension (mostlikly .raw). Don't forget to put it as raw string using r before the name : r'Checkerboard'.
    "registration_directory": r"",
}


# ---------------------------------------------------------------------------
# Paths that can differ per user / machine — EDIT these if your setup differs.
# (Leave a value as None to use the automatic default.)
# ---------------------------------------------------------------------------
# Spike-sorting output:
#   - If you ran the sorting with notebook 1, leave both as None: the Sorting folder
#     is <root>/Sorting and the phy ".GUI" folder inside it is found automatically.
#   - If you sorted on another machine / in another folder, set the path(s) explicitly.
sorting_directory_override = None  # e.g. r"/media/other_pc/exp/Sorting"
phy_directory_override = None  # e.g. r"/media/other_pc/exp/Sorting/recording_0/recording_0.GUI"

# ---------------------------------------------------------------------------
# Stimulus (.vec) files
# ---------------------------------------------------------------------------
# Folder holding the standard stimulus ".vec" files (chirp, drifting-gratings,
# cell-typing). The pipeline ships them in "RessourcesAndTools/StandardVec".
# If YOUR experiment used a different version of a stimulus (a differently-named .vec,
# possibly with different parameters), either drop that .vec into this folder, or point
# this path at wherever your .vec files live. When a step cannot find the .vec it expects
# here, it lists the .vec files in this folder and asks you to pick the right one.
stim_directory = r"./RessourcesAndTools/StandardVec"


# setup MEA parameters (always check!)
mea_params = {
    "mea_spacing": 30,  # the spacing between two electrodes of the MEA in µm for registration
    "n_electrodes": 16,  # number of electrodes on one side of the MEA. N tot electrodes = n_electrodes**2
}

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



########################################################
# Path Utils
########################################################
def make_dict_keys_global_variables(params: dict):
    for k, v in params.items():
        globals()[k] = v


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


def find_phy_directory(search_dirs):
    """Locate the phy export folder (name ends in '.GUI').

    Phy writes its arrays (spike_clusters.npy, spike_times.npy, ...) into a folder
    whose name ends in '.GUI'. Its name and location depend on how the sorting was run:
    it may sit under the Sorting folder, or next to the raw file in the recordings
    folder. So we search several directories (up to 3 levels deep) instead of hardcoding.

    Args:
        search_dirs: a directory, or a list of directories, to search.

    Returns:
        Path to the .GUI folder, or None if none is found. If several are found, the
        first is returned and the rest are listed.
    """
    if isinstance(search_dirs, str):
        search_dirs = [search_dirs]
    candidates = []
    for base in search_dirs:
        if base and os.path.isdir(base):
            candidates += glob.glob(os.path.join(base, "*.GUI"))
            candidates += glob.glob(os.path.join(base, "*", "*.GUI"))
            candidates += glob.glob(os.path.join(base, "*", "*", "*.GUI"))
    candidates = sorted({os.path.normpath(c) for c in candidates if os.path.isdir(c)})
    if not candidates:
        return None
    if len(candidates) > 1:
        print(
            "- /!\\ Several phy (.GUI) folders found; using the first. "
            "Set phy_directory_override in params.py to pick another:"
        )
        print(*[f"    {c}" for c in candidates], sep="\n")
    return candidates[0]


def create_path_automatically(params: dict):
    print("\n-------- Creating all paths ---------\n")

    # Link to the actual raw files frome the recording
    # listed in the input_file
    recording_directory = os.path.join(params["root"], params["raw_files_folder"])

    # Sorting folder (where spike-sorting / phy output lives). This is only a PATH:
    # it is NOT created here. You only need it if you run the sorting with this pipeline,
    # and that step creates it on demand — so users who sort elsewhere get no empty
    # "Sorting" folder. Override it if you sorted in another location.
    if sorting_directory_override:
        sorting_directory = os.path.normpath(sorting_directory_override)
    else:
        sorting_directory = os.path.join(params["root"], r"Sorting")
    # kept for backward compatibility (create_symlinks still expects this name)
    symbolic_link_directory = sorting_directory

    # phy ".GUI" folder (contains spike_clusters.npy, spike_times.npy, ...).
    # Resolution order:
    #   1. explicit override (phy_directory_override)
    #   2. auto-detected *.GUI folder, searched in BOTH the Sorting folder and the raw-
    #      recordings folder (spyking-circus often writes it next to the raw file)
    if phy_directory_override:
        phy_directory = os.path.normpath(phy_directory_override)
        ok = "" if os.path.isdir(phy_directory) else "   /!\\ (this path does not exist!)"
        print(f"- phy (.GUI): using your override{ok}\n    {phy_directory}")
    else:
        phy_directory = find_phy_directory([sorting_directory, recording_directory])
        if phy_directory is not None:
            print(f"- phy (.GUI): found automatically\n    {phy_directory}")
        else:
            phy_directory = os.path.join(sorting_directory, "recording_00", "recording_00.GUI")
            print(
                "- phy (.GUI): NOT found yet — no '.GUI' folder under the Sorting or the\n"
                "    raw-recordings folder. This is fine if you have not sorted yet.\n"
                "    If you HAVE sorted, set 'phy_directory_override' in params.py to your\n"
                "    .GUI folder (the one containing spike_clusters.npy)."
            )

    # Link to the directory where output data should be saved
    output_directory = os.path.join(params["root"], r"Analysis")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f'- Created "output" path: {output_directory}')
    else:
        print('- "output" path already exists')

    # Link to the folder in which triggers will be saved.
    # If doesn't exist, will be created.
    triggers_directory = os.path.join(output_directory, "triggers")
    if not os.path.exists(triggers_directory):
        os.makedirs(triggers_directory)
        print(f'- Created "triggers" path: {triggers_directory}')
    else:
        print('- "triggers" path already exists')

    # Path to the checkerboard binary file used to generate stimuli
    binary_source_path = os.path.join(ressources, "binarysource1000Mbits")

    raw_filtered_directory = os.path.join(params["root"], "RAW_filtered")

    registration_frames = os.path.join(
        params["root"], params["registration_directory"] + r"/frames/"
    )

    registration_imgs = os.path.join(
        params["root"], params["registration_directory"] + r"/imgs/"
    )

    # Do not use this unless you know how !!!
    if not os.path.exists(recording_directory):
        print(f'Creating "recording_directory" path: {recording_directory}')
        print("Please make sure to fill it with your raw files!")
        os.makedirs(recording_directory)

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
(
    recording_directory,  # Link to the actual raw files from the recording listed in the input_file
    symbolic_link_directory,
    sorting_directory,
    phy_directory,
    output_directory,  # Directory where preprocessing info are saved
    triggers_directory,  # folder in which the triggers are saved
    binary_source_path,
    raw_filtered_directory,
    registration_frames,
    registration_imgs,
    recording_names,  # Recordings labels available
) = create_path_automatically(basic_params)


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


# ---------------------------------------------------------------------------
# SWN (Sparse White Noise) stimulus — an alternative to the checkerboard.
# Used by 2-Analyse_Checkerboard.ipynb when is_swn = True.
# Give just the file NAMES: they are looked up in 'stim_directory' (StandardVec),
# exactly like the other stimulus files — and if a name is not found there, the folder
# is listed and you are asked to pick the matching file. The real SWN .bin can be very
# large; if you keep it outside the repo, put its full absolute path here instead of a
# bare file name (that works too). The MEA/rig id and DMD pixel size come from the MEA
# settings above. The name usually encodes the settings, e.g.
#   20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2  ->  48 px/check, 6 px shift, 30 Hz, MEA 2
# ---------------------------------------------------------------------------
swn_bin_file = "20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2.bin"
swn_vec_file = "20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2.vec"
swn_shift_x = 6  # spatial down-sampling step in x (pixels) — matches "6pixShift" in the stim design
swn_shift_y = 6  # spatial down-sampling step in y (pixels)
swn_cov_regularization = 5.0  # sigma added to the stimulus-covariance diagonal (stabilises STA whitening)
