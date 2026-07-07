"""Drifting-gratings (DG) analysis.

The stimulus shows 8 grating directions (45° apart), each repeated 4 times. This
module reuses the generic vec-based sequence extraction from ``utils`` (the same
functions the Standard_Vec_Analysis notebook uses) to split each cell's spikes per
direction and repetition, then computes direction tuning and plots rasters + polar
tuning curves.

contact: laquitainesteeve@gmail.com
"""

# import packages
import os
import itertools
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# import custom packages
import utils

# ==========================
# Stimulus constants
# ==========================

N_DIRECTIONS = 8  # number of grating directions (45° apart)
N_REPETITIONS = 4  # number of times each direction is shown

# Per grating speed: sweep duration of one grating and the separation used to lay
# the 8 directions side by side on a single plot time axis (purely for display).
DG_SPEED_SETTINGS = {
    0: {"label": "FAST", "seq_len": 3.96, "seq_sep": 9},
    1: {"label": "MEDIUM", "seq_len": 6.0, "seq_sep": 10},
    2: {"label": "SLOW", "seq_len": 12.0, "seq_sep": 20},
}

# The original pipeline subtracted no baseline firing rate.
BASELINE_FIRING = 0


def direction_key_to_angle_index(direction_key) -> int:
    """Map a vec direction key (1..8) to the angle index 0..7 used by the tuning code.

    The original pipeline indexed angles as ``7 - <grating index in presentation
    order>`` (the "counterclockwise" flip). The vec key encodes ``<grating index> + 1``,
    so the equivalent angle index is ``N_DIRECTIONS - direction_key``.

    If a later validation shows rotated/flipped tuning curves, the vec's direction
    ordering differs from the old hardcoded order and THIS is the single place to fix it.
    """
    return N_DIRECTIONS - int(direction_key)


# ==========================
# Loading utilities
# ==========================


def prompt_user_for_dg_speed():
    """Ask the user for the grating speed and return its settings.

    Returns
    -------
    seq_len : float
        Duration (s) of one grating sweep.
    seq_sep : float
        Separation (s) between gratings on the plot time axis (display only).
    label : str
        Human-readable speed label.
    """
    speed = int(input("\nSelect grating speed (0 = fast, 1 = medium, 2 = slow (default)): "))
    if speed not in DG_SPEED_SETTINGS:
        raise ValueError("Grating speed must be 0 (fast), 1 (medium) or 2 (slow = default).")
    settings = DG_SPEED_SETTINGS[speed]
    return settings["seq_len"], settings["seq_sep"], settings["label"]


def infer_n_repetitions_from_vec(vec_keys, n_digit_for_rep: int = 4) -> int:
    """Infer the number of repetitions per direction from the vec sequence keys.

    Each vec key is ``<direction><repetition>``; the last ``n_digit_for_rep`` digits are
    the 0-based repetition index. The number of repetitions is therefore the largest
    repetition index seen, plus one.
    """
    reps = np.asarray(vec_keys).astype(int) % (10**n_digit_for_rep)
    return int(reps.max()) + 1


def get_all_inputs_for_dg_analysis(params):
    """Prompt for the recording, vec file and speed, then load everything needed.

    The number of repetitions per direction is read from the vec and shown to you for
    confirmation (press Enter to accept, or type the correct number).

    Parameters
    ----------
    params : object
        Experiment parameters (from params.py).

    Returns
    -------
    cells : list[np.uint32]
        Cluster identifiers.
    spike_times : dict[int, np.ndarray]
        Spike times per cell (s).
    stim_onsets : np.ndarray
        Stimulus onset times (s), one per row of the vec file.
    vec_keys : np.ndarray
        Sequence key of each trigger (the vec file's last column).
    seq_len : float
        Duration of a grating sweep (s).
    seq_sep : float
        Separation between gratings on the plot time axis (s).
    DG_directory : str
        Output directory for the DG analysis.
    n_repetitions : int
        Number of repetitions per direction (read from the vec, confirmed by the user).
    """
    rec_idx, rec = utils.prompt_user_for_recording(params.recording_names, "DG recording")
    DG_directory = utils.create_analysis_directory(params.output_directory, rec_idx, "DG")

    seq_len, seq_sep, _ = prompt_user_for_dg_speed()

    # Vec file (its last column gives the direction+repetition key of every trigger).
    # Looks for the standard DG vec in params.stim_directory; if your experiment used a
    # different version, it lists the .vec files there and asks you to pick the right one.
    vec_path = utils.find_vec_file("DG_50hZ_8reps_8dir_2sT_std.vec", params.stim_directory)
    vec = np.loadtxt(vec_path)[1:, :]  # drop header line
    vec_keys = vec[:, -1]

    # Number of repetitions per direction: read from the vec, confirmed by the user.
    detected = infer_n_repetitions_from_vec(vec_keys, n_digit_for_rep=4)
    answer = input(
        f"\nDetected {detected} repetitions per direction from the vec. "
        "Press Enter to accept, or type the correct number: "
    ).strip()
    n_repetitions = int(answer) if answer else detected
    print(f"Using {n_repetitions} repetitions per direction.\n")

    triggers_path = os.path.normpath(
        os.path.join(params.triggers_directory, f"{params.exp}_{rec}_triggers.pkl")
    )
    stim_onsets = utils.load_stim_onset_from_triggers_path(
        triggers_path, params.fs, verbose=True
    )
    cells, spike_times = utils.load_spike_times(rec, params.output_directory, params.exp)

    return (
        cells,
        spike_times,
        stim_onsets,
        vec_keys,
        seq_len,
        seq_sep,
        DG_directory,
        n_repetitions,
    )


# ==========================
# DG analysis
# ==========================


def compute_dg_rasters(
    cells,
    spike_times,
    stim_onsets,
    vec_keys,
    seq_len,
    seq_sep,
    DG_directory,
    params,
    n_repetitions: int = N_REPETITIONS,
    n_digit_for_rep: int = 4,
):
    """Build per-direction rasters and direction tuning for every cell, then save them.

    Spikes are split per grating direction and repetition with the generic vec
    function ``utils.build_spikes_per_sequence_dict``. They are then laid out the way
    ``compute_tuning`` expects: the 8 directions placed side by side on a single
    time axis, one row per repetition, each direction offset by ``seq_sep``.

    ``n_repetitions`` is how many times each direction is shown in YOUR stimulus (the vec)
    — set it to match your experiment (e.g. 10 for a 10-rep DG). As in the original
    pipeline, the first repetition of each direction is dropped (repetitions
    1..n_repetitions-1 are used), the last raster row is left empty, and no baseline
    firing rate is subtracted.

    The result is saved as ``{cell_id: DG_data}`` to ``DG_data_exp<exp>.pkl`` in
    ``DG_directory`` (see ``compute_tuning`` for the contents of ``DG_data``).
    """
    # Per-direction, per-repetition spikes from the vec file:
    #   spikes_per_sequence[cell][direction_key]["raster"] = [rep0, rep1, rep2, rep3]
    # each repetition being the spike times referenced to that grating's onset.
    spikes_per_sequence, _, _ = utils.build_spikes_per_sequence_dict(
        cells, spike_times, stim_onsets, vec_keys, n_digit_for_rep=n_digit_for_rep
    )

    DG_set = {}
    for cell in tqdm(cells, desc="Computing direction selectivity"):
        directions = spikes_per_sequence[cell]

        # Lay the 8 directions side by side on one time axis, one row per repetition.
        # Drop repetition 0 to match the original pipeline: repetitions 1..3 go to rows
        # 0..2, and the 4th row stays empty.
        ch_raster = [[] for _ in range(n_repetitions)]
        for direction_key, sequence in directions.items():
            angle = direction_key_to_angle_index(direction_key)
            repetitions = sequence["raster"]  # [rep0, rep1, ..., rep(n_repetitions-1)]
            for row, rep in enumerate(range(1, n_repetitions)):  # drop rep 0
                ch_raster[row] = np.append(
                    ch_raster[row], repetitions[rep] + angle * seq_sep
                )

        if not list(itertools.chain(*ch_raster)):
            continue  # this cell fired no spikes during the stimulus

        *_, DG_data = compute_tuning(
            ch_raster, BASELINE_FIRING, seq_len, seq_sep, n_repeats=n_repetitions
        )
        DG_set[cell] = DG_data

    utils.save_obj(DG_set, os.path.join(DG_directory, f"DG_data_exp{params.exp}"))
    print("--- Done ---")


def plot_dg_rasters(DG_directory, seq_sep, seq_len, params, fontsize=16, show=False):
    """Plot and save, for every cell, its DG raster + PSTH and polar direction tuning.

    Reads the ``DG_data`` dictionary saved by ``compute_dg_rasters`` and writes one
    PNG per cell into ``DG_directory/DG_figs``.

    Parameters
    ----------
    DG_directory : str
        Directory containing the saved DG data; figures go in its ``DG_figs`` subfolder.
    seq_sep : float
        Separation between gratings on the time axis (s), as used when computing.
    seq_len : float
        Duration of a grating sweep (s).
    params : object
        Experiment parameters (from params.py).
    fontsize : int
        Base font size for titles and labels (ticks use ``fontsize - 2``).
    show : bool
        If True, also display each figure in the notebook.

    Returns
    -------
    matplotlib.figure.Figure
        The last figure created (handy to display the final cell in a notebook).
    """
    DG_set = utils.load_obj(os.path.join(DG_directory, f"DG_data_exp{params.exp}"))

    fig_directory = os.path.normpath(os.path.join(DG_directory, "DG_figs"))
    os.makedirs(fig_directory, exist_ok=True)

    direction_degrees = np.arange(0, 360, 360 // N_DIRECTIONS)  # 0, 45, ..., 315

    fig = None
    for cell in tqdm(DG_set.keys(), desc="Plotting"):
        data = DG_set[cell]
        n_rep = len(data["rasters"])  # repetitions used (matches compute_dg_rasters)

        fig = plt.figure(figsize=(12, 11))
        fig.suptitle(f"Cell {cell}", fontsize=fontsize + 4)
        gs = fig.add_gridspec(5, 3, hspace=0.6, wspace=0.3)

        # ---- Raster (one row per repetition) + PSTH, 8 directions side by side ----
        ax = fig.add_subplot(gs[0:2, :])
        ax.eventplot(data["rasters"], color="k", lw=1, linelengths=0.95)
        for a in range(N_DIRECTIONS):
            ax.axvline(a * seq_sep, color="lightgray", lw=1)  # grating onset
            ax.axvline(a * seq_sep + seq_len, color="lightgray", lw=1)  # grating offset
        # PSTH drawn above the raster rows (scaled to ~n_rep rows tall).
        psth = data["counts"] / data["maxcount"] * n_rep + (n_rep + 0.5)
        ax.hist(
            data["bins"][:-1],
            data["bins"],
            histtype="step",
            lw=1.5,
            color="darkblue",
            weights=psth,
        )
        ax.set_xlim([-seq_sep / 2, seq_sep * N_DIRECTIONS])
        ax.set_xticks([a * seq_sep + seq_len / 2 for a in range(N_DIRECTIONS)])
        ax.set_xticklabels([f"{d}°" for d in direction_degrees], fontsize=fontsize - 2)
        ax.set_yticks(range(n_rep))
        ax.tick_params(axis="y", labelsize=fontsize - 2)
        ax.set_xlabel("Direction", fontsize=fontsize)
        ax.set_ylabel("Repetition", fontsize=fontsize)
        ax.set_title("Raster + PSTH per direction", fontsize=fontsize)

        # ---- Direction tuning (single polar plot) ----
        ax = fig.add_subplot(gs[2:5, 1], polar=True)
        theta = np.linspace(0, 2 * np.pi, N_DIRECTIONS + 1)
        ax.plot(theta, data["Tuning"], color="#B85A8F", lw=2.5)
        ax.fill(theta, data["Tuning"], color="#B85A8F", alpha=0.2)
        ax.plot(
            [data["atune"], data["atune"]], [0, data["Rtune"]], color="k", lw=2
        )  # preferred direction
        ax.plot([data["atune"]], [data["Rtune"]], "ko")
        ax.set_thetagrids(direction_degrees, fontsize=fontsize - 2)
        ax.set_ylim([0, 1])
        ax.set_yticks([0.5, 1])
        ax.set_yticklabels(["0.5", "1"], fontsize=fontsize - 4)
        ax.set_title(
            f"Direction tuning\nIDX = {data['IDX']:.2f}    R = {data['Rtune']:.2f}",
            fontsize=fontsize,
            pad=25,
        )

        # ---- Save (and optionally show) ----
        fig_path = os.path.join(
            fig_directory, f"DG_resp_exp{params.exp}_Cell_{cell}.png"
        )
        if show:
            plt.show(block=False)
        fig.savefig(fig_path, dpi=90, bbox_inches="tight")
        plt.close(fig)

    print("--- Done ---")
    return fig

# ==========================
# Compute Tuning
# ==========================

def _dg_bin_params(seq_sep):
    """Time-binning constants shared by the DG PSTH and the per-direction responses."""
    nbins = 8 * 10 * 20  # total nb of bins (1600)
    binsize = seq_sep * 8 * 1000 // nbins  # bin size in ms
    binsec = 1000 // binsize  # nb bins per second
    bins = np.linspace(0, seq_sep * 8, nbins + 1)
    return nbins, binsize, binsec, bins


def compute_dg_psth(ch_raster, seq_sep, base_fire=0, n_repeats=4):
    """PSTH of the drifting-gratings raster: the 8 directions binned side by side.

    Returns the histogram ``counts`` (baseline subtracted), its peak ``maxcount`` and the
    bin edges ``bins``.
    """
    merged = list(itertools.chain(*ch_raster))  # all directions' spikes on one axis
    nbins, _, _, bins = _dg_bin_params(seq_sep)
    base_fire = base_fire * (seq_sep * 8 / nbins) * n_repeats
    counts, bins = np.histogram(merged, bins=bins)
    counts = counts - base_fire
    maxcount = np.amax(counts)
    return counts, maxcount, bins


def compute_direction_responses(counts, seq_len, seq_sep):
    """Total response per grating direction (spikes in the response window of each angle).

    Returns a length-9 vector (index 8 repeats index 0), un-normalised. For each angle the
    response is summed from ``seq_len / 6`` after onset to the grating offset.
    """
    _, binsize, binsec, _ = _dg_bin_params(seq_sep)
    tune = np.zeros(9)
    for a in np.arange(8):
        sel_bins = np.copy(
            counts[
                int(seq_len * 1000 / 6) // binsize
                + int(seq_sep * binsec * a) : int(seq_len * binsec + seq_sep * binsec * a)
            ]
        )
        tune[a] = np.sum(sel_bins)
        if a == 0:
            tune[a + 8] = np.sum(sel_bins)
    return tune


def compute_tuning_vector(tune):
    """Preferred direction and vector strength from a per-direction response.

    ``tune`` is the un-normalised length-9 response vector. Returns the normalised tuning
    vector, the preferred-direction angle ``atune`` (radians) and the vector strength ``R``.
    """
    vxs = vys = 0
    for a in np.arange(8):
        vxs += np.cos(np.pi * a * 45 / 180) * tune[a]
        vys += np.sin(np.pi * a * 45 / 180) * tune[a]
    peak = np.amax(tune)
    vxs = vxs / peak
    vys = vys / peak
    tune_norm = tune / peak
    atune = np.arctan2(vys, vxs)
    R = np.sqrt(vys**2 + vxs**2)
    return tune_norm, atune, R


def compute_dsi(tune_norm, atune):
    """Direction-selectivity index from the normalised tuning and the preferred angle.

    The arithmetic (including the retry logic when the index is very negative) is preserved
    exactly from the original pipeline.
    """
    tune = tune_norm[:-1]

    def dsi(angle):
        opposite = int((angle + 4) % 8)
        return (tune[angle] - tune[opposite]) / (tune[angle] + tune[opposite])

    angle = int(np.round(atune / np.pi * 4))
    idx = dsi(angle)
    if idx < -0.2:
        angle2 = angle + 1
        idx = dsi(angle2)
        angle = angle2
    if idx < -0.2:
        angle2 = angle - 2
        idx = dsi(angle2)
        if idx < -0.2:
            angle = angle + 1
        idx = dsi(angle)
    return idx


def compute_tuning(ch_raster, base_fire, seq_len, seq_sep, n_repeats=4):
    """Compute direction tuning of one cell from its drifting-gratings raster.

    Args:
        ch_raster: List of repetition rows. Each row holds the spike times of all
            directions laid side by side on one time axis: direction ``a`` (0..7) is
            offset by ``a * seq_sep`` and referenced to its grating onset.
        base_fire: Baseline firing rate to subtract (spikes/s). 0 keeps raw counts.
        seq_len: Grating sweep duration (s); spikes from ``seq_len / 6`` to ``seq_len``
            after onset are counted as the response per direction.
        seq_sep: Separation between directions on the time axis (s).
        n_repeats: Number of repetitions represented in ``ch_raster``.

    Returns:
        TuneSum: Per-direction normalised tuning (9 values; index 8 repeats index 0).
        atune: Preferred direction angle (radians).
        R: Tuning vector strength.
        IDX: Direction-selectivity index.
        counts, maxcount, bins: PSTH histogram of ``ch_raster`` and its peak/bin edges.
        DG_data: Dict bundling all of the above plus ``rasters`` (the input ``ch_raster``).

    Note:
        The arithmetic here is preserved exactly from the original pipeline (including
        its time-binning) so results stay reproducible; only documentation was added.
    """
    # Each metric is computed by its own small function (see above); this just chains
    # them and bundles the result. The arithmetic is unchanged from the original pipeline.
    counts, maxcount, bins = compute_dg_psth(ch_raster, seq_sep, base_fire, n_repeats)
    tune = compute_direction_responses(counts, seq_len, seq_sep)

    if sum(tune) == 0:  # cell with no response -> original placeholder output
        DG_data = {
            "IDX": 0,
            "Tuning": tune,
            "atune": 0,
            "Rtune": 0,
            "rasters": np.zeros((4, len(bins))),
            "counts": counts,
            "maxcount": maxcount,
            "bins": bins,
        }
        return np.zeros(9), 0, 0, 0, counts, maxcount, bins, DG_data

    tune_norm, atune, R = compute_tuning_vector(tune)
    IDX = compute_dsi(tune_norm, atune)

    DG_data = {
        "IDX": IDX,
        "Tuning": tune_norm,
        "atune": atune,
        "Rtune": R,
        "rasters": ch_raster,
        "counts": counts,
        "maxcount": maxcount,
        "bins": bins,
    }
    return tune_norm, atune, R, IDX, counts, maxcount, bins, DG_data