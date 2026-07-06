import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import *  # noqa: F401, F403
from matplotlib.gridspec import GridSpec

import utils
import params

# Chirp vec files (one sequence repeated many times). The "_std" versions carry the
# per-repetition sequence keys added by RessourcesAndTools/StimMaking/add_standard_keys_to_chirp_vec.ipynb.
CHIRP_VEC_FILES = {
    False: "Euler_50Hz_20reps_1024x768pix_std.vec",  # new 50 Hz chirp
    True: "EulerStim180530_std.vec",  # old 2p-room chirp (needs its keyed vec generated)
}


def get_all_inputs_for_chirp_analysis(params: dict, old: bool):
    """Load triggers, spikes and the chirp vec keys, then build the analysis directories.

    Args:
        params: Experiment parameters from params.py (exp, recording_names,
            output_directory, triggers_directory, fs, root).
        old (bool): If True, use the old 2p-room chirp; if False, the new 50 Hz chirp.

    Returns:
        cells (list[np.uint32]): neuron cluster IDs.
        spike_times (dict[int, np.ndarray]): spike times per cell (s).
        stim_onsets (np.ndarray): stimulus onset times (s), aligned with the vec rows.
        vec_keys (np.ndarray): sequence key of each trigger (the chirp vec's last column).
        check_directory (str): checkerboard analysis directory.
        CT_directory (str): cell typing output directory.
        old (bool): chirp type flag (passed through).

    Note:
        Prompts the user to select the recording. Creates the cell typing directory.
    """

    # Prompt user to select recording
    recording_number, rec = utils.prompt_user_for_recording(
        params.recording_names, "chirp recording"
    )
    print(f"\nSelected recording : {rec} \n")

    # Create cell typing directory
    CT_directory = utils.create_analysis_directory(
        params.output_directory, recording_number, "CellTyping"
    )

    check_directory = utils.find_analysis_directory(
        params.output_directory, dir_type="Checkerboard"
    )

    # Load triggers
    triggers_path = os.path.normpath(
        os.path.join(params.triggers_directory, f"{params.exp}_{rec}_triggers.pkl")
    )
    stim_onsets = utils.load_stim_onset_from_triggers_path(
        triggers_path, params.fs, verbose=True
    )

    # Load spike trains
    cells, spike_times = utils.load_spike_times(rec, params.output_directory, params.exp)
    print(f"Total : {len(spike_times)} neurons loaded \n\nClusters id :\n{cells}\n")

    # Load the chirp vec keys (last column) used to split spikes per repetition.
    vec_filename = CHIRP_VEC_FILES[old]
    vec_path = os.path.join(params.stim_directory, vec_filename)
    vec_keys = np.loadtxt(vec_path)[1:, -1]  # drop header row
    print(f"Chirp vec keys loaded : {vec_filename}")

    return (
        cells,
        spike_times,
        stim_onsets,
        vec_keys,
        check_directory,
        CT_directory,
        old,
    )


def compute_chirp_rasters(
    cells: list,
    spike_times: dict,
    stim_onsets: np.ndarray,
    vec_keys: np.ndarray,
    old: bool = False,
    n_bins: int = 800,
    n_bins_small: int = 16000,
    n_digit_for_rep: int = 4,
):
    """Compute chirp responses (rasters, PSTH, noise) per cell, split per repetition.

    Repetition boundaries come from the chirp vec's sequence keys (the chirp is one
    sequence type repeated many times) via ``utils.group_triggers_by_sequence`` — the
    same engine as the other vec-based analyses. This replaces the previous hardcoded
    trigger slicing while keeping the binning identical.

    Args:
        cells (list): cell/cluster IDs.
        spike_times (dict[int, np.ndarray]): spike times per cell (s).
        stim_onsets (np.ndarray): trigger times (s), aligned with the vec rows.
        vec_keys (np.ndarray): sequence key of each trigger (the chirp vec's last column).
        old (bool): if True, old-chirp binning (n_bins=625, rep length 25 s); else new (32 s).
        n_bins (int): number of PSTH bins.
        n_bins_small (int): number of fine bins for the noise analysis.
        n_digit_for_rep (int): trailing key digits encoding the repetition.

    Returns:
        dict: cell_data[cell_id] with keys 'repeated_sequences_times', 'spike_trains',
            'psth' (Hz), 'mean_spikes_count_small_bin', 'spikes_counts_small_bin',
            'noise_small_bin', 'noise_large_bin'.
    """

    # Processing-------------------------------------------
    print("Extracting cells responses to Chirp stimulus\n")

    if old:
        n_bins = 625  # Change here for old chirp n_bins
        rep_lenght = 25
    else:
        rep_lenght = 32
    time_bin = rep_lenght / n_bins  # in seconds

    # Repetition boundaries from the vec: one trigger list per repetition key.
    # Keys with empty sequence-type part (the "0" lead-in/trailing group) are dropped.
    triggers_per_repetition = utils.group_triggers_by_sequence(stim_onsets, vec_keys)
    rep_keys = sorted(
        key for key in triggers_per_repetition if key[:-n_digit_for_rep] != ""
    )
    nb_repetitions = len(rep_keys)

    # [start, end] of each repetition (identical for every cell).
    repeated_sequences_times = [
        [triggers_per_repetition[key][0], triggers_per_repetition[key][-1]]
        for key in rep_keys
    ]

    cell_data = {}
    for cell_nb in tqdm(cells, desc="Extraction"):
        euler_sptimes = spike_times[cell_nb]

        # Spike train of each repetition, aligned to its start.
        spike_trains = []
        for start, end in repeated_sequences_times:
            spike_trains.append(utils.restrict_array(euler_sptimes, start, end) - start)

        # Bin at two timescales: coarse for the PSTH, fine for the noise analysis.
        binned_spikes = np.empty((nb_repetitions, n_bins))
        spike_counts_small_bin = np.empty((nb_repetitions, n_bins_small))
        for i in range(nb_repetitions):
            binned_spikes[i, :] = np.histogram(
                spike_trains[i], bins=n_bins, range=(0, rep_lenght)
            )[0]
            spike_counts_small_bin[i, :] = np.histogram(
                spike_trains[i], bins=n_bins_small, range=(0, rep_lenght)
            )[0]

        psth = np.sum(binned_spikes, axis=0)
        mean_spikes_count = np.sum(spike_counts_small_bin, axis=0) / nb_repetitions

        cell_data[cell_nb] = {
            "repeated_sequences_times": repeated_sequences_times,
            "spike_trains": spike_trains,
            "psth": psth / time_bin,
            "mean_spikes_count_small_bin": mean_spikes_count,
            "spikes_counts_small_bin": spike_counts_small_bin,
            "noise_small_bin": np.subtract(spike_counts_small_bin, mean_spikes_count),
            "noise_large_bin": np.subtract(binned_spikes, psth / rep_lenght),
        }

    return cell_data


def plot_chirp_rasters(
    cells: list,
    cell_data: dict,
    CT_directory: str,
    check_directory: str,
    old: bool = False,
    fontsize: int = 16,
):
    """Generate and save chirp raster plots for all cells.

    Each figure stacks the chirp stimulus, the spike raster and the PSTH on the left,
    with the cell's spatial STA shown large on the right (full height).

    Args:
        cells (list): List of cell/cluster IDs to plot
        cell_data (dict): Dictionary containing chirp response data for all cells
        CT_directory (str): Path to cell typing output directory
        check_directory (str): Path to checkerboard analysis directory containing STA results
        old (bool): If True, use old chirp parameters. Default False.
        fontsize (int): Base font size for titles and labels (ticks use fontsize - 2).

    Returns:
        None. Saves PNG files to CT_directory/Chirp_rasters+STA/
    """

    # Input-------------------------------------------------------------
    fig_directory = os.path.normpath(os.path.join(CT_directory, r"Chirp_rasters+STA"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    sta_results = np.load(
        os.path.join(check_directory, "sta_data_analysed_extended.pkl"), allow_pickle=True
    )

    if old:
        vec_path = os.path.join(params.stim_directory, r"EulerStim180530.vec")
        euler_vec = -np.genfromtxt(vec_path)
        rep_lenght = 25
        n_bins = 625

    else:
        vec_path = os.path.join(params.stim_directory, r"Euler_50Hz_20reps_1024x768pix.vec")
        euler_vec = np.genfromtxt(vec_path)
        rep_lenght = 32
        n_bins = 800

    # Processing------------------------------------------------------------

    print(f"Saving Chirp raster plots in : {fig_directory} \n")

    time_bin = rep_lenght / n_bins  # in seconds
    for cell_nb in tqdm(cells[:]):
        fig = plt.figure(figsize=(18, 8))
        # Left column (cols 0-15): stimulus / raster / PSTH stacked and sharing the x-axis.
        # Right column (cols 17-23): the spatial STA, spanning the full height.
        gs = GridSpec(
            8,
            24,
            left=0.07,
            right=0.97,
            bottom=0.1,
            top=0.92,
            wspace=0.5,
            hspace=0.0,
            figure=fig,
        )

        # --- Chirp stimulus (top) ---
        ax_stim = fig.add_subplot(gs[0:2, :16])
        if old:
            ax_stim.plot(
                np.linspace(0, rep_lenght, 999), euler_vec[1:1000, 1], color="k", lw=1.5
            )
        else:
            ax_stim.plot(
                np.linspace(0, rep_lenght, 1600),
                euler_vec[151:1751, 1],
                color="k",
                lw=1.5,
            )
        ax_stim.set_ylabel("Stimulus", fontsize=fontsize)
        ax_stim.set_yticks([])
        ax_stim.set_title(f"Cluster {cell_nb}", fontsize=fontsize + 4)
        ax_stim.set_xlim([0, rep_lenght])

        # --- Spike raster ---
        ax_rast = fig.add_subplot(gs[2:5, :16], sharex=ax_stim)
        ax_rast.eventplot(
            cell_data[cell_nb]["spike_trains"], color="k", lw=1, linelengths=1
        )
        ax_rast.set_ylabel("Trial", fontsize=fontsize)
        ax_rast.tick_params(axis="y", labelsize=fontsize - 2)

        # --- PSTH (bottom, the only panel with the time axis) ---
        ax_psth = fig.add_subplot(gs[5:8, :16], sharex=ax_stim)
        ax_psth.step(
            np.linspace(0, rep_lenght, n_bins),
            cell_data[cell_nb]["psth"],
            color="#1f5fb0",
            lw=1.5,
        )
        ax_psth.set_ylabel(
            f"Firing rate (Hz)\n{int(time_bin * 1000)} ms bins", fontsize=fontsize
        )
        ax_psth.set_xlabel("Time (s)", fontsize=fontsize)
        ax_psth.set_xlim([0, rep_lenght])
        ax_psth.tick_params(labelsize=fontsize - 2)

        # Declutter: hide the redundant x tick labels on the top two panels (shared x),
        # and drop the top/right (and stimulus left) spines.
        plt.setp(ax_stim.get_xticklabels(), visible=False)
        plt.setp(ax_rast.get_xticklabels(), visible=False)
        for ax in (ax_stim, ax_rast, ax_psth):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_stim.spines["left"].set_visible(False)

        # --- Spatial STA (right, full height) ---
        ax_sta = fig.add_subplot(gs[:, 17:])
        ax_sta.set_title("STA", fontsize=fontsize + 2)
        spatial = sta_results[cell_nb]["sta_analysis"]["Spatial"]
        spatial = spatial**2 * np.sign(spatial)
        image = ax_sta.imshow(spatial, cmap="RdBu_r", interpolation="gaussian")
        abs_max = 0.5 * max(np.max(spatial), abs(np.min(spatial)))
        image.set_clim(-abs_max, abs_max)
        ax_sta.set_xticks([])
        ax_sta.set_yticks([])

        fsave = os.path.join(fig_directory, "{}_Chirp_raster+STA".format(cell_nb))
        fig.savefig(fsave + ".png", format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)

    print("--- Cell Done ---")

    return
