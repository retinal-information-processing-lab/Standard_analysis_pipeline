"""Module of functions for the drifting gratings analysis"""

"""
Drifting Gratings (DG) analysis module

This module provides utilities to:
- Load trigger times and spike trains
- Compute drifting grating rasters and tuning metrics
- Plot DG rasters and tuning curves

Author: laquitainesteeve@gmail.com
"""

# ==========================
# Imports
# ==========================
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import utils


# ==========================
# Loading utilities
# ==========================


def prompt_user_for_dg_speed():
    """
    Prompt user to select drifting grating speed.

    Returns
    -------
    seq_len : float
        Duration (s) of one grating sweep.
    seq_sep : float
        Temporal separation (s) between gratings (for plotting).
    trigsinrep : int
        Number of trigger samples per repetition.
    ttext : str
        Human-readable speed label.
    """
    T = int(input("\nSelect Grating's speed (T=0 fast, T=1 medium, T=2 slow) : "))

    if T == 0:
        return 3.96, 9, int(50 * 3.96), "FAST"
    if T == 1:
        return 6, 10, 50 * 6, "MEDIUM"
    if T == 2:
        return 12, 20, 50 * 12, "SLOW"

    raise ValueError("Invalid grating speed selection.")


def get_all_inputs_for_dg_analysis(params):
    """
    High-level loader for DG analysis.

    This function:
    - Prompts user for recording and speed
    - Loads trigger onsets
    - Loads spike trains
    - Prepares output directories

    Parameters
    ----------
    params : object
        Experiment parameters.

    Returns
    -------
    cells : list[np.uint32]
        Cluster identifiers.
    spike_times : list[np.ndarray]
        Spike times per cluster.
    stim_onsets : np.ndarray
        Stimulus onset times (s).
    trigsinrep : int
        Number of trigger samples per repetition.
    seq_sep : float
        Separation between gratings (s).
    seq_len : float
        Duration of a grating (s).
    DG_directory : str
        Output directory for DG analysis.
    """
    rec_idx, rec = utils.prompt_user_for_recording(params, "DG recording")
    DG_directory = utils.create_analysis_directory(
        params.output_directory, rec_idx, "DG"
    )

    seq_len, seq_sep, trigsinrep, _ = prompt_user_for_dg_speed()
    stim_onsets, _ = utils.load_triggers(params, rec)
    cells, spike_times = utils.load_spike_trains(params, rec)

    return cells, spike_times, stim_onsets, trigsinrep, seq_sep, seq_len, DG_directory


# ==========================
# Raster computation
# ==========================


def get_dg_sequence():
    """
    Return drifting grating angle sequence.

    Returns
    -------
    np.ndarray
        Angle indices (0–7) for 32 gratings.
    """
    DG_seq = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        4,
        1,
        5,
        2,
        0,
        3,
        7,
        6,
        1,
        4,
        0,
        3,
        2,
        5,
        6,
        7,
        5,
        2,
        3,
        6,
        1,
        4,
        7,
        0,
    ]

    return (np.ones(32) * 7 - DG_seq).astype(int)


def compute_dg_rep_starts(stim_onsets, trigsinrep):
    """
    Compute stimulus repetition start times.

    Parameters
    ----------
    stim_onsets : np.ndarray
        Trigger onset times.
    trigsinrep : int
        Triggers per repetition.

    Returns
    -------
    list of float
        Start times of each grating sweep.
    """
    nb_rep = len(stim_onsets) // trigsinrep
    return [stim_onsets[n * trigsinrep] for n in range(nb_rep)]


def build_ch_raster(spike_times, dg_rep_starts, DG_seq, seq_len, seq_sep):
    """
    Build raster structure across repetitions and angles.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times for one neuron.
    dg_rep_starts : list
        Start times for each grating.
    DG_seq : np.ndarray
        Angle index per grating.
    seq_len : float
        Grating duration.
    seq_sep : float
        Artificial separation between gratings.

    Returns
    -------
    list[list[float]]
        Raster data (4 repetitions).
    """
    n_angles = 8
    ch_raster = [[] for _ in range(4)]
    dg_count = np.zeros(n_angles, dtype=int)

    for n in range(8, len(dg_rep_starts)):
        t0 = dg_rep_starts[n]
        t1 = t0 + seq_len if n == len(dg_rep_starts) - 1 else dg_rep_starts[n + 1]

        rep_sptimes = spike_times[(spike_times > t0) & (spike_times < t1)]
        angle = DG_seq[n]
        rep = dg_count[angle]

        ch_raster[rep] = np.append(ch_raster[rep], rep_sptimes - t0 + angle * seq_sep)
        dg_count[angle] += 1

    return ch_raster


def compute_dg_rasters(
    cells, spike_times, stim_onsets, trigsinrep, seq_len, seq_sep, DG_directory, params
):
    """
    Compute drifting grating rasters and tuning metrics.

    Results are saved as a pickle file.

    Parameters
    ----------
    cells : list[np.uint32]
        Cluster identifiers.
    spike_times : list[np.ndarray]
        Spike times per cluster.
    stim_onsets : np.ndarray
        Stimulus onset times.
    trigsinrep : int
        Triggers per repetition.
    seq_len : float
        Grating duration.
    seq_sep : float
        Grating separation.
    DG_directory : str
        Output directory.
    params : object
        Experiment parameters.
    """
    DG_seq = get_dg_sequence()
    DG_set = {}

    dg_rep_starts = compute_dg_rep_starts(stim_onsets, trigsinrep)

    for i, clus in enumerate(tqdm(cells, desc="Computing Direction Selectivity")):
        ch_raster = build_ch_raster(
            spike_times[i], dg_rep_starts, DG_seq, seq_len, seq_sep
        )

        if not list(itertools.chain(*ch_raster)):
            continue

        base_fire = 0  # baseline firing rate (not estimated here)

        (TuneSum, atune, R, IDX, counts, maxcount, bins, DG_data) = (
            utils.compute_tuning(ch_raster, base_fire, seq_len, seq_sep)
        )

        DG_set[clus] = DG_data

    savef = os.path.join(DG_directory, f"DG_data_exp{params.exp}")
    utils.save_obj(DG_set, savef)
    print("--- Cell Done ---")


# ==========================
# Plotting
# ==========================
def plot_dg_cell(DG_set, cell, seq_sep, seq_len, fig_directory, exp, show=False):
    """
    Plot DG rasters and tuning for a single cell.
    Parameters
    ----------
    DG_set : dict
        DG data for all cells.
    cell : np.uint32
        Cluster identifier.
    seq_sep : float
        Grating separation.
    seq_len : float
        Grating duration.
    fig_directory : str
        Directory to save figures.
    exp : int
        Experiment identifier.
    show : bool, optional
        Whether to display the figure, by default False.
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # --------plot the rasters-------------------
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Cell {}".format(cell))

    gs = fig.add_gridspec(
        5, 8, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.7
    )

    ax = fig.add_subplot(gs[0:2, 0:8])
    utils.plot_single_raster(ax, DG_set[cell]["rasters"][:])
    for a in np.arange(8):
        ax.axvline(a * seq_sep, color="gray", lw=2)
        ax.axvline(a * seq_sep + seq_len, color="gray", lw=2)
        ax.axvline(a * seq_sep + seq_len / 6, color="gray", ls="--", lw=1.5)

    ax.set_xlim([-seq_sep / 2, seq_sep * 8])
    ax.set_ylim([-0.5, 3.5 + 2 + 4 + 2])
    ax.set_yticks(np.arange(4))
    ax.set_ylabel("Repetition               Counts       ", size=10)
    ax.set_xlabel("Time (s) {8 angles}", size=10)
    # fig.suptitle(ttext+'    cluster '+str(clus) + '      '+'% spikes: ' +str(round(len(dg_sptimes)/len(sp_times)*100,1))+'    Nspikes '+str(Nspikes))
    plt.rc(
        "axes.spines", **{"bottom": False, "left": False, "right": False, "top": False}
    )
    ax.text(
        5,
        12,
        "0                      45                      90                    135                    180                   225                    270                   315",
    )
    ax.axhline(3.5 + 2, color="k", lw=0.5)  # base_firing

    # --------------------------plot the histograms------------------------
    counts = DG_set[cell]["counts"] / DG_set[cell]["maxcount"] * 4 + 3.5 + 2
    ax.hist(
        DG_set[cell]["bins"][:-1],
        DG_set[cell]["bins"],
        histtype="step",
        lw=1.5,
        color="darkblue",
        weights=counts,
    )

    # --------------------------plot the polar plot left--------------

    ax = fig.add_subplot(gs[2:5, 1:4], polar=True)

    theta = np.linspace(0, 2 * np.pi, 9)
    # Arrange the grid into number of sales equal parts in degrees
    lines, labels = plt.thetagrids(range(0, 360, int(360 / 8)), np.arange(0, 360, 45))

    # Plot actual sales graph
    ax.plot(theta, DG_set[cell]["Tuning"])
    ax.fill(theta, DG_set[cell]["Tuning"], "b", alpha=0.1)
    #         ax.plot(theta, TuneMax,'orange')

    ax.plot(
        [DG_set[cell]["atune"], DG_set[cell]["atune"]], [0, DG_set[cell]["Rtune"]], "b-"
    )
    ax.plot([DG_set[cell]["atune"]], [DG_set[cell]["Rtune"]], "bo")

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([])
    ax.set_ylim([0, 1])

    ax.text(
        np.pi * 1 / 5, 1.3, "IDX = " + str(np.round(DG_set[cell]["IDX"], 1)), size=18
    )
    ax.text(
        np.pi * 1 / 8, 1.25, "R = " + str(np.round(DG_set[cell]["Rtune"], 1)), size=18
    )

    # ---------------------------plot the polar plot right (same as left but not limited between 0 and 1)------
    ax = fig.add_subplot(gs[2:5, 5:8], polar=True)

    ax.plot(
        [DG_set[cell]["atune"], DG_set[cell]["atune"]], [0, DG_set[cell]["Rtune"]], "b-"
    )
    ax.plot([DG_set[cell]["atune"]], [DG_set[cell]["Rtune"]], "bo")
    ax.plot(theta, DG_set[cell]["Tuning"])
    ax.fill(theta, DG_set[cell]["Tuning"], "b", alpha=0.1)

    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels([0, "", 1, "", 2])

    # -----------------------------------------------------------------------------------
    fsave = os.path.join(fig_directory, "DG_resp_exp{}_Cell_{}".format(exp, cell))
    if show:
        print(cell)
        plt.show(block=False)
    return fig


def plot_dg_rasters(DG_directory, seq_sep, seq_len, params):
    """
    Plot DG rasters and tuning curves for each cell.

    Parameters
    ----------
    DG_directory : str
        Directory containing DG data.
    seq_sep : float
        Grating separation.
    seq_len : float
        Grating duration.
    params : object
        Experiment parameters.

    Returns
    -------
    matplotlib.figure.Figure
        Last generated figure.
    """
    DG_set = utils.load_obj(os.path.join(DG_directory, f"DG_data_exp{params.exp}"))

    fig_directory = os.path.join(DG_directory, "DG_figs")
    os.makedirs(fig_directory, exist_ok=True)

    for cell in tqdm(DG_set.keys(), desc="Plotting"):
        fig = plot_dg_cell(
            DG_set[cell], cell, seq_sep, seq_len, fig_directory, params.exp
        )
        plt.close(fig)

    print("--- Cell Done ---")
    return fig
