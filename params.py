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
import warnings

# setup pipeline parameters
# relative path from pipeline notebook to a folder containing resources such as mea pictures and datasets
resources = r"./resources"

# setup experiment parameters (always check!)

basic_params = {
    "root": r"/media/idv-s8/SSD Storage/20260729_VideoLSTA_calib/",  # This is the root folder of your experiment; all other files must be inside of this folder or manually specified.
    "exp": r"20260729_VideoLSTA_calib",  # name of your experiment for saving the triggers
    "MEA": 2,  # select MEA (3=2p room) (4=MEA1 Polychrome)
    "raw_files_folder": r"RAW_DATA",  # Enter the name of the folder containing all your raw files. It will be conctenated with root to find your raws. If the folder is not in root, change the variable "recording_directory" manually.
    "recording_names": [
        "20260729_meas_00_SWN_30Hz",
        "20260729_meas_01_SWN_30Hz",
        "20260729_meas_02_chirp_50Hz",
        "20260729_meas_03_DG_2ST_10rep_50Hz",
        "20260729_meas_04_barcode_3dir_50Hz",
        "20260729_meas_05_ RMO_Pert_CALIB_baseline_40Hz",
        "20260729_meas_06_RMO_Pert_CALIB_c15_f04_40Hz",
        "20260729_meas_07_RMO_Pert_CALIB_c15_f12_40Hz",
        "20260729_meas_08_RMO_Pert_CALIB_c20_f04_40Hz_2026",
        "20260729_meas_09_RMO_Pert_CALIB_c20_f12_40Hz",
        "20260729_meas_10_RMO_Pert_CALIB_baseline_40Hz",
        "20260729_meas_11_RMO_Pert_CALIB_c25_f04_40Hz",
        "20260729_meas_12_RMO_Pert_CALIB_c25_f12_40Hz_2026",
        "20260729_meas_13_RMO_Pert_CALIB_c30_f04_40Hz",
        "20260729_meas_14_RMO_Pert_CALIB_c30_f12_40Hz",
        "20260729_meas_15_RMO_Pert_CALIB_baseline_40Hz",
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
phy_directory_override = "/media/idv-s8/SSD Storage/20260729_VideoLSTA_calib/RAW_DATA/20260729_meas_00_SWN_30Hz/20260729_meas_00_SWN_30Hz.GUI"

# ---------------------------------------------------------------------------
# Stimulus (.vec) files
# ---------------------------------------------------------------------------
# Folder holding the standard stimulus ".vec" files (chirp, drifting-gratings,
# cell-typing). The pipeline ships them in "ResourcesAndTools/StandardVec".
# If YOUR experiment used a different version of a stimulus (a differently-named .vec,
# possibly with different parameters), either drop that .vec into this folder, or point
# this path at wherever your .vec files live. When a step cannot find the .vec it expects
# here, it lists the .vec files in this folder and asks you to pick the right one.
stim_directory = r"./ResourcesAndTools/StandardVec"


# setup MEA parameters (always check!)
mea_params = {
    "mea_spacing": 30,  # the spacing between two electrodes of the MEA in µm for registration
    "n_electrodes": 16,  # number of electrodes on one side of the MEA. N tot electrodes = n_electrodes**2
}

# setup advanced parameters
# Default values used in utils functions. If a function has a wrong behaviour, you may want to look in here.
advanced_params = {
    "dtype": "uint16",  # Datatype used to open rawfiles recordings
    # voltage_resolution moved to most_advanced_params: it is now DERIVED per-rig from the
    # amplifier gain + input range in rig_params (see voltage_resolution_uV), not a constant.
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


# ---------------------------------------------------------------------------
# Rig (MEA) hardware settings — the SINGLE SOURCE OF TRUTH for everything rig-dependent.
# The code reads these values instead of hard-coding them, so adding or fixing a rig is
# done HERE and nowhere else.
#
# Per rig:
#   amplifier_gain     recording amplifier gain (dimensionless)
#   input_range_mV     ADC input range, ± full-scale referred to the input, in mV
#                      -> the voltage resolution (µV per ADC level) is DERIVED from these
#                         two plus ADC_N_BITS, so it is not stored separately (see
#                         voltage_resolution_uV). This is what makes recorded amplitudes
#                         physical, and lets the same physical trigger threshold be compared
#                         across rigs.
#   threshold_uV       trigger-detection threshold, in physical µV. Its SIGN is the trigger
#                      pulse polarity (positive-going vs negative-going on that rig).
#   size_dmd           dimensions of the DMD, in pixels [x, y]
#   pxl_size_dmd       size of one DMD pixel, in µm
#   max_frame_size     largest stimulus image the rig can display [x, y], in pixels
#                      (used to sanity-check a .bin before reading/writing it)
#   invert_polarity    True if the rig displays inverted, so frames are stored inverted
#                      and must be flipped back (value -> 1 - value) when read
#   optical_transform  geometric correction compensating this rig's optical path, applied
#                      when reading/writing a stimulus .bin. One of:
#                        "rot90_flipud", "fliplr", or None (no correction)
#
# Only MEA 2 and 3 are implemented AND tested for stimulus display. Using another rig is
# allowed but is WORK IN PROGRESS: you get a loud warning, and anything left as None falls
# back to a neutral default (no optical correction, no polarity inversion, no size check),
# so frames may come out mirrored / rotated / inverted. To properly support a rig, fill in
# its values below and list it in DISPLAY_READY_RIGS.
# ---------------------------------------------------------------------------
# Number of bits the ADC uses to encode the amplified signal (why the raw dtype is uint16).
ADC_N_BITS = 16

rig_params = {
    1: {
        "amplifier_gain": 1,
        "input_range_mV": 4096,
        "threshold_uV": 3530000,  # = legacy 270000 scaled units
        "size_dmd": [760, 1020],
        "pxl_size_dmd": 3.5,
        "max_frame_size": [1024, 768],
        "invert_polarity": False,
        "optical_transform": "rot90_flipud",
    },
    2: {
        "amplifier_gain": 1,
        "input_range_mV": 4096,
        "threshold_uV": 1920000,  # = legacy 150000 scaled units
        "size_dmd": [864, 864],
        "pxl_size_dmd": 3.5,
        "max_frame_size": [1920, 1080],
        "invert_polarity": False,
        "optical_transform": "rot90_flipud",
    },
    3: {
        "amplifier_gain": 1,
        "input_range_mV": 4096,
        "threshold_uV": 2222000,  # = legacy 170000 scaled units
        "size_dmd": [760, 1020],
        "pxl_size_dmd": 2.5,
        "max_frame_size": [1024, 768],
        "invert_polarity": True,
        "optical_transform": "fliplr",
    },
    4: {
        # Electrical meta unknown so far -> voltage resolution falls back to legacy (warns).
        "amplifier_gain": None,
        "input_range_mV": None,
        "threshold_uV": -3414.402,  # = legacy -314470 scaled units #WRONG
        "size_dmd": None,
        "pxl_size_dmd": None,
        "max_frame_size": None,
        "invert_polarity": None,
        "optical_transform": None,
    },
    5: {
        # Electrical meta unknown so far -> voltage resolution falls back to legacy (warns).
        "amplifier_gain": None,
        "input_range_mV": None,
        "threshold_uV": -76.004,  # = legacy -7000 scaled units #WRONG
        "size_dmd": [760, 1020],
        "pxl_size_dmd": 3.5,
        # Display settings unknown so far -> stimulus .bin reading/writing warns (see below).
        "max_frame_size": None,
        "invert_polarity": None,
        "optical_transform": "rot90_flipud",
    },
}

# Rigs whose stimulus-display settings (DMD geometry + optics) are implemented and tested.
DISPLAY_READY_RIGS = (1, 2, 3)


def get_rig_params(mea: int) -> dict:
    """All hardware settings of one rig (see rig_params above)."""
    if mea not in rig_params:
        raise ValueError(
            f"MEA {mea} is not defined in params.rig_params "
            f"(known rigs: {sorted(rig_params)}). Add its settings there."
        )
    return rig_params[mea]


def get_display_rig_params(mea: int) -> dict:
    """Rig settings needed to read/write a stimulus .bin (DMD geometry + optics).

    Rigs outside DISPLAY_READY_RIGS are WORK IN PROGRESS: they have not been implemented
    or tested. Rather than blocking you, this warns loudly and falls back to whatever is
    filled in for that rig, with neutral defaults for what is missing (no optical
    correction, no polarity inversion, no frame-size limit). The frames you get may then
    be mirrored, rotated or inverted — only rely on this if you know what you are doing.
    """
    settings = get_rig_params(mea)
    if mea not in DISPLAY_READY_RIGS:
        missing = [
            k
            for k in ("max_frame_size", "invert_polarity", "optical_transform")
            if settings[k] is None
        ]
        warnings.warn(
            f"MEA {mea}: reading/writing stimulus .bin files is WORK IN PROGRESS — this rig "
            f"has not been implemented or tested (tested rigs: {list(DISPLAY_READY_RIGS)}).\n"
            f"  Settings not defined for it in params.rig_params: {missing or 'none'}.\n"
            "  Falling back to neutral defaults for those: no optical correction, no polarity "
            "inversion, no frame-size check.\n"
            "  The frames you read/write may therefore be mirrored, rotated or inverted. Only "
            "rely on this if you know what you are doing — and once you know this rig's real "
            "values, fill them in in params.rig_params and add it to params.DISPLAY_READY_RIGS.",
            stacklevel=2,
        )
        settings = {
            **settings,
            # neutral defaults for anything this rig does not define
            "invert_polarity": bool(settings["invert_polarity"]),  # None -> False
            "optical_transform": settings["optical_transform"],  # None -> identity
            "max_frame_size": settings["max_frame_size"],  # None -> no size check
        }
    return settings


# Voltage resolution the pipeline used before it was derived per-rig (µV per ADC level).
# Kept only as the fallback for rigs whose electrical meta is not filled in yet.
LEGACY_VOLTAGE_RESOLUTION_UV = 0.1042


def voltage_resolution_uV(mea: int) -> float:
    """µV per ADC level for a rig, DERIVED from its amplifier gain and input range.

        resolution = full input span referred to the electrode / number of ADC levels
                   = (2 * input_range_mV / gain) * 1000 / 2**ADC_N_BITS      [µV / level]

    (``input_range_mV`` is the ± full-scale, so the full span is ``2 * input_range_mV``.)
    For rigs 1/2/3 (gain 1200, ±4096 mV, 16 bits) this gives ≈ 0.1042 µV/level.

    Rigs whose electrical meta is not filled in yet fall back to the legacy value with a
    warning, so the pipeline still runs but the amplitudes are not physically calibrated.
    """
    settings = get_rig_params(mea)
    gain, input_range_mV = settings["amplifier_gain"], settings["input_range_mV"]
    if gain is None or input_range_mV is None:
        warnings.warn(
            f"MEA {mea}: 'amplifier_gain' / 'input_range_mV' not set in params.rig_params, so "
            f"the voltage resolution cannot be derived. Falling back to the legacy "
            f"{LEGACY_VOLTAGE_RESOLUTION_UV} µV/level. Fill in this rig's electrical meta to make "
            "the recorded amplitudes (and the trigger threshold) physically correct.",
            stacklevel=2,
        )
        return LEGACY_VOLTAGE_RESOLUTION_UV
    full_span_uV = (2 * input_range_mV / gain) * 1000
    return full_span_uV / (2**ADC_N_BITS)


def setup_threshold_pxl_size_size_dmd(params: dict):
    """Trigger-detection threshold (µV), DMD pixel size (µm) and DMD dimensions of this rig.

    All values come from the rig_params table above (they vary with the rig).
    """
    settings = get_rig_params(params["MEA"])
    return settings["threshold_uV"], settings["pxl_size_dmd"], settings["size_dmd"]


most_advanced_params = {
    "threshold": setup_threshold_pxl_size_size_dmd(basic_params)[
        0
    ],  # trigger threshold, µV
    "voltage_resolution": voltage_resolution_uV(
        basic_params["MEA"]
    ),  # µV per ADC level
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

    # Every path below is always DEFINED, but folders are only CREATED when the experiment
    # root actually exists. Without the data (a fresh clone, another machine, the automated
    # tests) "import params" must still work — the steps that really need the data then fail
    # later with a clear message. It also means a typo in 'root' no longer silently creates
    # a whole bogus folder tree somewhere.
    root_exists = os.path.isdir(params["root"])
    if not root_exists:
        print(
            f"- /!\\ Experiment root folder NOT found:\n    {params['root']}\n"
            "    Paths are still defined, but no folder was created and no recording listed.\n"
            "    Set 'root' in params.py to your experiment folder before running an analysis.\n"
        )

    def make_dir(path, label):
        """Create 'path' (only if the experiment root exists) and report what happened."""
        if not root_exists:
            return path
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'- Created "{label}" path: {path}')
        else:
            print(f'- "{label}" path already exists')
        return path

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
        ok = (
            ""
            if os.path.isdir(phy_directory)
            else "   /!\\ (this path does not exist!)"
        )
        print(f"- phy (.GUI): using your override{ok}\n    {phy_directory}")
    else:
        phy_directory = find_phy_directory([sorting_directory, recording_directory])
        if phy_directory is not None:
            print(f"- phy (.GUI): found automatically\n    {phy_directory}")
        else:
            phy_directory = os.path.join(
                sorting_directory, "recording_00", "recording_00.GUI"
            )
            print(
                "- phy (.GUI): NOT found yet — no '.GUI' folder under the Sorting or the\n"
                "    raw-recordings folder. This is fine if you have not sorted yet.\n"
                "    If you HAVE sorted, set 'phy_directory_override' in params.py to your\n"
                "    .GUI folder (the one containing spike_clusters.npy)."
            )

    # Link to the directory where output data should be saved
    output_directory = make_dir(os.path.join(params["root"], r"Analysis"), "output")

    # Link to the folder in which triggers will be saved.
    # If doesn't exist, will be created.
    triggers_directory = make_dir(
        os.path.join(output_directory, "triggers"), "triggers"
    )

    # Path to the checkerboard binary file used to generate stimuli
    binary_source_path = os.path.join(resources, "binarysource1000Mbits")

    raw_filtered_directory = os.path.join(params["root"], "RAW_filtered")

    registration_frames = os.path.join(
        params["root"], params["registration_directory"] + r"/frames/"
    )

    registration_imgs = os.path.join(
        params["root"], params["registration_directory"] + r"/imgs/"
    )

    # Do not use this unless you know how !!!
    if root_exists and not os.path.exists(recording_directory):
        print(f'Creating "recording_directory" path: {recording_directory}')
        print("Please make sure to fill it with your raw files!")
        os.makedirs(recording_directory)

    # No raw-files folder (no data on this machine) -> no recordings to list.
    recording_names = (
        find_files(recording_directory) if os.path.isdir(recording_directory) else []
    )
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
# Trigger / auxiliary channels.
# A few of the MEA's channels do not record neural signal but experiment metadata: the
# stimulus-onset trigger and a few auxiliary channels (holographic trigger, laser-shutter
# state, colour code...). Preprocessing now ALWAYS saves the raw trace of every channel
# listed here, per recording (as float32 µV), so their information is available for sanity
# checks without having to re-extract it — and so the extraction no longer needs to know
# the recording type.
# Give {name: channel_id}; the names become the keys of the saved data. The onset detection
# still runs on params.visual_channel_id (the "trigger_channel" below).
trigger_channel_ids = {
    "trigger_channel": most_advanced_params[
        "visual_channel_id"
    ],  # 126 — stimulus-onset trigger
    "aux_channel_1": most_advanced_params[
        "holo_channel_id"
    ],  # 127 — (was the holographic trigger)
    "aux_channel_2": 254,
    "aux_channel_3": 255,
}


# ---------------------------------------------------------------------------
# SWN (Shifting White Noise) stimulus — an alternative to the checkerboard.
# Used by 2-Analyse_Checkerboard.ipynb when is_swn = True.
# Like the checkerboard, each SWN sequence is a novel (never-repeated) first half followed
# by a repeated half: 1200 frames = 600 novel + 600 repeated (20 s at 30 Hz), x 45 reps.
# The novel half gives the STA, the repeated half gives the rasters / reliability.
# The raw .vec has no sequence keys; ResourcesAndTools/StimMaking/add_standard_keys_to_swn_vec.ipynb
# writes a "*_std.vec" with them, usable by 6_Standard_Vec_Analysis.ipynb.
# Each of the two settings below accepts EITHER of:
#   * a bare file NAME  -> looked up in 'stim_directory' (StandardVec), like the other
#     stimulus files; the folder is listed and you are asked to confirm which file to use.
#   * a full PATH       -> used as-is, with no prompt. The SWN .bin is very large, so keep
#     it wherever it already lives (e.g. an external drive) instead of duplicating it into
#     StandardVec. A path starting with "~" works too.
# The MEA/rig id and DMD pixel size come from the MEA settings above. The name usually
# encodes the settings, e.g.
#   20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2  ->  48 px/check, 6 px shift, 30 Hz, MEA 2
# Example of a heavy .bin kept outside the repo:
#   swn_bin_file = r"/media/my_drive/Stimuli/20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2.bin"
# ---------------------------------------------------------------------------
swn_bin_file = "20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2.bin"
swn_vec_file = "20250512_4_SWN_48pixCh_6pixShift_30Hz_MEA2_std.vec"
swn_shift_x = 6  # spatial down-sampling step in x (pixels) — matches "6pixShift" in the stim design
swn_shift_y = 6  # spatial down-sampling step in y (pixels)
swn_cov_regularization = (
    5.0  # sigma added to the stimulus-covariance diagonal (stabilises STA whitening)
)
