import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import *
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA, SparsePCA
import scipy as sc
from sklearn.cluster import AgglomerativeClustering

# Above is just everything from original utils.  Likely overkill but prevents annoying errors from not having a needed module.


import utils

"""Custom functions###################################################################

[TODO!]: 
- convert each cell of the notebook to functions X
- describe function arguments' data type 0
- describe functions with docstrings /
- move functions to utils for this notebook.py X
"""


# Chirp vec files (one sequence repeated many times). The "_std" versions carry the
# per-repetition sequence keys added by ressources/add_standard_keys_to_chirp_vec.ipynb.
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
    recording_number, rec = utils.prompt_user_for_recording(params, "chirp recording")
    print(f"\nSelected recording : {rec} \n")

    # Create cell typing directory
    CT_directory = utils.create_analysis_directory(
        params, recording_number, "CellTyping"
    )

    check_directory = utils.find_analysis_directory(dir_type="Checkerboard")

    # Load triggers
    triggers_path = os.path.normpath(
        os.path.join(params.triggers_directory, f"{params.exp}_{rec}_triggers.pkl")
    )
    stim_onsets = utils.load_stim_onset_from_triggers_path(
        triggers_path, params, verbose=True
    )

    # Load spike trains
    cells, spike_times = utils.load_spike_times(params, rec)
    print(f"Total : {len(spike_times)} neurons loaded \n\nClusters id :\n{cells}\n")

    # Load the chirp vec keys (last column) used to split spikes per repetition.
    vec_filename = CHIRP_VEC_FILES[old]
    vec_path = os.path.join("./ressources", vec_filename)
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
        os.path.join(check_directory, "sta_data_3D_fitted.pkl"), allow_pickle=True
    )

    if old:
        vec_path = os.path.join("./ressources", r"EulerStim180530.vec")
        euler_vec = -np.genfromtxt(vec_path)
        rep_lenght = 25
        n_bins = 625

    else:
        vec_path = os.path.join("./ressources", r"Euler_50Hz_20reps_1024x768pix.vec")
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
        spatial = sta_results[cell_nb]["center_analyse"]["Spatial"]
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


def select_and_save_cells_for_clustering(
    cells,
    good_sta_cells: list,
    good_chirp_cells: list,
    CT_directory: str,
    check_directory: str,
    params: dict,
):
    """Update cell selection for clustering analysis.

    Args:
        good_sta_cells (list): Cell IDs with good STA quality (or empty list)
        good_chirp_cells (list): Cell IDs with good chirp responses (or empty list)
        CT_directory (str): Path to cell typing output directory
        check_directory (str): Path to the checkerboard analysis directory (its
            ``Stas_figs`` subfolder holds the STA figures used to judge STA quality)
        params (dict): Experiment parameters containing 'exp' field

    Returns:
        tuple: Contains:
            - selected_cells (list): Final list of all selected cells
            - selected_cells_sta (list): Cells selected based on STA quality
            - selected_cells_chirp (list): Cells selected based on chirp quality

    Note:
        Calls cell_selection_for_clustering() which should be defined in utils.
        This function provides an interactive interface for manual cell selection.
    """

    # Input-------------------------------------------------

    exp = params.exp

    fig_directory = os.path.normpath(os.path.join(CT_directory, r"Chirp_rasters+STA"))
    # Path to the file saving the cells to use for clustering
    all_selected_cells_file = os.path.normpath(
        os.path.join(CT_directory, "{}_selected_cells_for_clustering.pkl".format(exp))
    )

    if os.path.isfile(all_selected_cells_file):
        print(f"Loading previous selection from  : {all_selected_cells_file}")
        all_selected_cells = utils.load_obj(all_selected_cells_file)
        selected_cells = all_selected_cells["selected_cells"]
        selected_cells_sta = all_selected_cells["selected_cells_sta"]
        selected_cells_chirp = all_selected_cells["selected_cells_chirp"]
    else:
        selected_cells = []
        selected_cells_sta = []
        selected_cells_chirp = []

    # Processing-------------------------------------------------

    if good_sta_cells:
        selected_cells_sta = good_sta_cells
    if good_chirp_cells:
        selected_cells_chirp = good_chirp_cells

    selected_cells, selected_cells_sta, selected_cells_chirp = (
        utils.cell_selection_for_clustering(
            cells,
            CT_directory_path=fig_directory,
            sta_figures_path=os.path.join(check_directory, "Stas_figs"),
            selected_cells_sta=selected_cells_sta,
            selected_cells_chirp=selected_cells_chirp,
        )
    )

    print("Selected {} cells.".format(len(selected_cells)))

    return selected_cells, selected_cells_sta, selected_cells_chirp


def modify_cells_for_clustering(
    cells,
    selected_cells_sta: list,
    selected_cells_sta_to_add: list,
    selected_cells_sta_to_remove: list,
    selected_cells_chirp: list,
    selected_cells_chirp_to_add: list,
    selected_cells_chirp_to_remove: list,
    remove_any_way: list,
    CT_directory: str,
    check_directory: str,
    params: dict,
):
    """Modify selected cells by adding/removing specific cells and save updated selection.

    Args:
        selected_cells_sta (list): Current STA-selected cells
        selected_cells_sta_to_add (list): Cell IDs to add to STA selection
        selected_cells_sta_to_remove (list): Cell IDs to remove from STA selection
        selected_cells_chirp (list): Current chirp-selected cells
        selected_cells_chirp_to_add (list): Cell IDs to add to chirp selection
        selected_cells_chirp_to_remove (list): Cell IDs to remove from chirp selection
        remove_any_way (list): Cell IDs to remove from all selections
        CT_directory (str): Path to cell typing output directory
        params (dict): Experiment parameters containing 'exp' field

    Returns:
        tuple: Contains:
            - selected_cells (list): Final combined selection
            - selected_cells_sta (list): Updated STA-selected cells
            - selected_cells_chirp (list): Updated chirp-selected cells

    Note:
        Saves updated selection to '{exp}_selected_cells_for_clustering.pkl'
        Function to review entirely completely
    """

    exp = (
        params.exp
    )  # Otherwise it can think that exp means the built in function exp not the experiment from params.

    # 2026-01-22 Leaving for now but this looks like a typo.  First line seems like it should be selected_cells_chirp and second selected_cells_sta, not both _sta
    selected_cells_sta = list(
        set(
            [
                idx
                for idx in selected_cells_chirp + selected_cells_chirp_to_add
                if idx not in selected_cells_chirp_to_remove + remove_any_way
            ]
        )
    )
    selected_cells_sta = list(
        set(
            [
                idx
                for idx in selected_cells_sta + selected_cells_sta_to_add
                if idx not in selected_cells_sta_to_remove + remove_any_way
            ]
        )
    )

    selected_cells, selected_cells_sta, selected_cells_chirp = (
        utils.cell_selection_for_clustering(
            cells,
            CT_directory_path=os.path.join(CT_directory, "Chirp_rasters+STA"),
            sta_figures_path=os.path.join(check_directory, "Stas_figs"),
            selected_cells_sta=list(set(selected_cells_sta)),
            selected_cells_chirp=list(set(selected_cells_chirp)),
        )
    )

    fsave = os.path.join(CT_directory, "{}_selected_cells_for_clustering".format(exp))
    utils.save_obj(
        {
            "selected_cells": selected_cells,
            "selected_cells_sta": selected_cells_sta,
            "selected_cells_chirp": selected_cells_chirp,
        },
        fsave,
    )

    return selected_cells, selected_cells_sta, selected_cells_chirp


def select_direction_selective_cells(
    selected_cells: list,
    ds_cells: list,
    DG_directory: str,
    CT_directory: str,
    params: dict,
):
    """Partition the clustering-selected cells into direction-selective (DS) and non-DS.

    The clustered cells are split in two so that the two groups can be cell-typed
    independently. Selection is either manual (pass a non-empty ``ds_cells`` list) or
    interactive: each cell's drifting-gratings figure (from the DG analysis, notebook 3)
    is shown and you confirm whether it is direction selective.

    Args:
        selected_cells (list): cells chosen for clustering (good STA + chirp).
        ds_cells (list): DS cell IDs to use directly; if empty, select interactively
            (or reload a previously saved selection).
        DG_directory (str): the DG analysis directory (its ``DG_figs`` holds the figures).
            Get it with ``utils.find_analysis_directory("DG")``.
        CT_directory (str): cell typing output directory (the selection is saved here).
        params (dict): experiment parameters containing 'exp'.

    Returns:
        ds_cells (list): direction-selective cells (a subset of selected_cells).
        non_ds_cells (list): the remaining selected cells.
    """
    exp = params.exp
    ds_file = os.path.normpath(
        os.path.join(CT_directory, f"{exp}_direction_selective_cells.pkl")
    )

    # Reuse a previously saved selection if none was given.
    if not ds_cells and os.path.isfile(ds_file):
        print(f"Loading previous DS selection from : {ds_file}")
        ds_cells = utils.load_obj(ds_file)["ds_cells"]

    # Otherwise ask the user, showing each cell's DG figure. Each figure replaces the
    # previous one (no endless scrolling) and is shown large enough to read.
    if not ds_cells:
        from IPython.display import clear_output

        dg_fig_directory = os.path.normpath(os.path.join(DG_directory, "DG_figs"))
        print("Selecting direction-selective cells from the DG plots ...")
        ds_cells = []
        for i, cell_nb in enumerate(selected_cells):
            fig_path = os.path.join(
                dg_fig_directory, f"DG_resp_exp{exp}_Cell_{cell_nb}.png"
            )
            if not os.path.isfile(fig_path):
                print(f"No DG figure for cell {cell_nb}, skipping.")
                continue
            clear_output(wait=True)  # remove the previous cell's figure + prompt
            print(f"DS selection — cell {i + 1}/{len(selected_cells)}")
            plt.figure("Current cell", figsize=(12, 11))
            plt.imshow(np.asarray(plt.imread(fig_path)))
            plt.axis("off")
            plt.show()
            if input(
                f"Is cell {cell_nb} direction selective? Type Yes to select : "
            ) in ["Y", "Yes", "y", "yes"]:
                ds_cells.append(cell_nb)
            plt.close("all")

    # Keep only DS cells that are actually in the clustering set; the rest are non-DS.
    ds_cells = [cell for cell in selected_cells if cell in ds_cells]
    non_ds_cells = [cell for cell in selected_cells if cell not in ds_cells]

    utils.save_obj(
        {"ds_cells": ds_cells, "non_ds_cells": non_ds_cells},
        os.path.join(CT_directory, f"{exp}_direction_selective_cells"),
    )
    print(f"{len(ds_cells)} direction-selective, {len(non_ds_cells)} non-DS cells.")

    return ds_cells, non_ds_cells


def run_cell_typing_AC(
    dist_thres: float,
    n_components_psth: int,
    n_components_sta_tc: int,
    cell_data: dict,
    selected_cells: list,
    check_directory: str,
    sparse: bool = False,
):
    """Run cell typing using Agglomerative Clustering on PSTH and STA features.

    Args:
        dist_thres (float): Distance threshold for clustering. Adjust to get ~50 clusters.
        n_components_psth (int): Number of PCA components for chirp PSTH.
                                Should explain ~80% variance.
        n_components_sta_tc (int): Number of PCA components for STA temporal component.
                                  Usually 2 for reliable checkerboard, 1 otherwise, 0 to skip.
        cell_data (dict): Dictionary containing spike train data for all cells
        selected_cells (list): List of cell IDs to include in clustering
        check_directory (str): Path to checkerboard analysis directory with STA results
        sparse (bool): If True, use SparsePCA for PSTH. Default False.

    Returns:
        tuple: Contains:
            - psth_z (np.ndarray): Z-scored PSTHs, shape (n_kept_cells, n_time_bins)
            - sta_results (dict): Loaded STA analysis results
            - model: Fitted AgglomerativeClustering model with cluster labels
            - kept_cells (list): selected_cells minus any dropped for a flat/NaN PSTH
              or STA; ``model.labels_[i]`` corresponds to ``kept_cells[i]``.

    Note:
        Cells whose chirp PSTH is flat (silent) or whose STA temporal course is flat
        or NaN are dropped (with a printed message), since z-scoring them produces NaN
        that PCA rejects. Use the returned kept_cells to map labels back to cells.
        Displays diagnostic plots: PCA variance, dendrogram, cluster centroids.
        You want ~80% cumulative variance explained by PCA components.
        Adjust dist_thres to get approximately 50 clusters.
    """
    # Input---------------------------------------------------

    sta_results = np.load(
        os.path.join(check_directory, "sta_data_3D_fitted.pkl"), allow_pickle=True
    )

    # Processing-----------------------------------------------------------

    # Drop cells whose features would be NaN: a flat (silent) chirp PSTH or a flat/NaN
    # STA temporal course both break the per-cell z-scoring used below, which PCA rejects.
    valid_cells, dropped = [], []
    for cell_id in selected_cells:
        sta_tc = np.asarray(
            sta_results[cell_id]["center_analyse"]["Temporal"][-21:], dtype=float
        )
        psth_ok = np.std(cell_data[cell_id]["psth"]) > 0
        sta_ok = np.all(np.isfinite(sta_tc)) and np.std(sta_tc) > 0
        (valid_cells if psth_ok and sta_ok else dropped).append(cell_id)
    if dropped:
        print(
            f"Dropping {len(dropped)} cell(s) with a flat/NaN PSTH or STA "
            f"(cannot be clustered): {dropped}"
        )
    selected_cells = valid_cells

    n_cells = len(selected_cells)
    # -----------------------------------
    # -----------------------------------
    # Get Euler PCA
    n_rep = 20  # nb of repeats
    nt = 32  # total length
    dt = 0.04  # bin size in seconds
    time_bins = np.arange(0, nt + dt, dt)

    # Bining
    spikes = np.zeros((n_cells, int(nt / dt), n_rep))
    for cell_index in range(len(selected_cells)):
        cell_id = selected_cells[cell_index]
        spike_cell = cell_data[cell_id]["spike_trains"]
        for rep in range(n_rep):
            temp = np.histogram(spike_cell[rep], bins=time_bins)
            spikes[cell_index, :, rep] = temp[0]

    # -------------------------
    # Pre process the PSTH
    psth = np.mean(spikes, 2)
    psth_z = sc.stats.zscore(psth, 1)

    if sparse:
        pca_transformer = SparsePCA(n_components_psth, random_state=0).fit(psth_z)
    else:
        pca_transformer = PCA(n_components_psth).fit(psth_z)
    psth_pca = pca_transformer.transform(psth_z)

    # -----------------------------------
    # Get checkerboard STA PCA
    STA_time_course = np.zeros((n_cells, 21))  # 21 data points for these STAs
    for cell_index in range(len(selected_cells)):
        cell_id = selected_cells[cell_index]
        TempSTA_cell = sta_results[selected_cells[cell_index]]["center_analyse"][
            "Temporal"
        ][-21:]
        STA_time_course[cell_index] = TempSTA_cell

    # ---------------------------
    # Pre process the STA
    sta_tc = sc.stats.zscore(STA_time_course[:, :], 1)

    if n_components_sta_tc > 0:
        pca_transformer2 = PCA(n_components_sta_tc).fit(sta_tc)
        sta_tc_pca = pca_transformer2.transform(sta_tc)

    # -----------------------------------
    cluster_dataset = np.zeros((n_cells, n_components_psth + n_components_sta_tc + 1))
    cluster_dataset[:, :n_components_psth] = psth_pca
    if n_components_sta_tc > 0:
        cluster_dataset[
            :, n_components_psth : n_components_psth + n_components_sta_tc
        ] = sta_tc_pca

    ell_size = np.zeros(len(selected_cells))
    for cell_index in range(len(selected_cells)):
        cell_id = selected_cells[cell_index]
        width, height = [
            sta_results[selected_cells[cell_index]]["center_analyse"]["EllipseCoor"][3],
            sta_results[selected_cells[cell_index]]["center_analyse"]["EllipseCoor"][4],
        ]
        #     width,height = cell_data[cell_id]["ellipseSigmaXY"]
        ell_size[cell_index] = np.abs(np.pi * width * height)

    ell_size_temp = -np.ones(n_cells)
    temp = ell_size[:] - ell_size[:].min()
    ell_size_temp[:] = temp / temp.max()
    cluster_dataset[:, -1] = ell_size_temp

    # -----------------------------------
    # perform agglomerative clustering
    model = AgglomerativeClustering(distance_threshold=dist_thres, n_clusters=None)
    # model = model.fit(psth_pca)
    model = model.fit(cluster_dataset)

    # Plotting------------------------------------------

    # Plot cumlative explained variance
    if not sparse:
        # For chirp PCAs
        plt.plot(
            np.arange(n_components_psth) + 1,
            np.cumsum(pca_transformer.explained_variance_ratio_) * 100,
        )
        plt.axhline(y=80, color="k")
        plt.xlabel("number of PCs from Chirp PSTH")
        plt.ylabel("% of cumulative explained variance")
        plt.show()

        if n_components_sta_tc > 0:
            # For STA PCAs
            plt.plot(
                np.arange(n_components_sta_tc) + 1,
                np.cumsum(pca_transformer2.explained_variance_ratio_) * 100,
                "o-",
            )
            plt.axhline(y=80, color="k")
            plt.xlabel("number of PCs from STA")
            plt.ylabel("% of cumulative explained variance")
            plt.show()

    # plot the dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    utils.plot_dendrogram(model, truncate_mode="level", p=0)
    plt.axhline(dist_thres, color="k")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    # plot the cluster centroids
    n_clusts = len(np.unique(model.labels_))
    plt.figure()
    for iclust in range(n_clusts):
        idx_cluster = np.where(model.labels_ == iclust)[0]
        plt.plot(np.mean(psth_z[idx_cluster, :], 0) + iclust * 5)
    plt.show()

    print("Number of clusters: ", len(np.unique(model.labels_)))

    # # plot the psths of all cells in one cluster
    # for icluster in range(len(np.unique(model.labels_))):
    #     # icluster = 0
    #     idx_cluster = np.where(model.labels_==icluster)[0]
    #     print(f'cluster size : {len(idx_cluster)}')
    #     plt.figure()
    #     plt.plot(psth_z[idx_cluster,:].transpose())
    #     plt.show()

    return psth_z, sta_results, model, selected_cells


def create_cluster_summary_figure(
    cell_data: dict,
    selected_cells: list,
    psth_z: np.ndarray,
    sta_results: dict,
    params: dict,
    CT_directory: str,
    old: bool,
    fontsize: int = 16,
    rf_zoom: int = 10,
):
    """Create one summary figure per cluster (robust to missing per-cell data).

    Each figure shows, per cell: orientation tuning (from the DG analysis), spatial and
    temporal STA and chirp PSTH; plus per-cluster summaries (RF-ellipse overlay, mean
    temporal STA, mean chirp PSTH, stimulus trace). Any missing piece is replaced by a
    "missing" note in the figure and a printed warning, instead of raising.

    Args:
        cell_data (dict): chirp response data for all cells.
        selected_cells (list): cells actually clustered; psth_z rows are aligned to this list.
        psth_z (np.ndarray): z-scored PSTHs, shape (len(selected_cells), n_time_bins).
        sta_results (dict): STA analysis results per cell.
        params (dict): experiment parameters ('exp').
        CT_directory (str): cell typing output directory (figures go in its Cell_typing/ subfolder).
        old (bool): if True, use the old chirp stimulus vec; else the new one.
        fontsize (int): base font size for titles/labels.
        rf_zoom (int): half-width (in STA pixels) of the spatial-STA window around the RF center.

    Returns:
        None. Saves one figure per cluster to CT_directory/Cell_typing/.
    """
    exp = params.exp

    fig_directory = os.path.normpath(os.path.join(CT_directory, "Cell_typing"))
    os.makedirs(fig_directory, exist_ok=True)

    # Optional inputs: warn and continue (with placeholders) if they cannot be loaded.
    DG_set = {}
    try:
        DG_set = utils.load_obj(
            os.path.join(utils.find_analysis_directory(dir_type="DG"), f"DG_data_exp{exp}")
        )
    except Exception as err:
        print(f"Warning: could not load DG tuning data ({err}); orientation plots skipped.")

    euler_vec = None
    try:
        vec_name = "EulerStim180530.vec" if old else "Euler_50Hz_20reps_1024x768pix.vec"
        euler_vec = np.genfromtxt(os.path.join("./ressources", vec_name))
        if old:
            euler_vec = -euler_vec
    except Exception as err:
        print(f"Warning: could not load chirp stimulus vec ({err}); stimulus trace skipped.")

    def missing(ax, message):
        """Blank an axis and write a small 'missing' note in it."""
        ax.axis("off")
        ax.text(
            0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize - 4, color="gray", style="italic",
        )

    cluster_ids = sorted(
        {cell_data[c]["type"] for c in cell_data if cell_data[c]["type"] != "Not assigned"}
    )

    for icluster in tqdm(cluster_ids, desc="Cluster summary figures"):
        cluster_cells = [c for c in selected_cells if cell_data[c].get("type") == icluster]
        n_cells = len(cluster_cells)
        print(f"Cluster {icluster}: {n_cells} cells")

        fig = plt.figure(figsize=(16, (n_cells + 2) * 1.9), constrained_layout=True)
        gs = fig.add_gridspec(
            n_cells + 2, 8, width_ratios=[1, 1, 1, 0.6, 1, 1, 1, 1]
        )
        fig.suptitle(
            f"Cell group {icluster} — {n_cells} cells", fontsize=fontsize + 4, fontweight="bold"
        )

        ax_ellipses = fig.add_subplot(gs[0:2, 1:3])

        temporal_sum = np.zeros(21)
        temporal_count = 0

        for row, cell_nb in enumerate(cluster_cells, start=2):

            # --- Orientation tuning (polar, from DG) ---
            ax = fig.add_subplot(gs[row, 0], polar=True)
            if cell_nb in DG_set:
                dg = DG_set[cell_nb]
                theta = np.linspace(0, 2 * np.pi, len(dg["Tuning"]))
                ax.plot(theta, dg["Tuning"], "b")
                ax.fill(theta, dg["Tuning"], "b", alpha=0.1)
                ax.plot([dg["atune"], dg["atune"]], [0, dg["Rtune"]], "b-")
                ax.plot([dg["atune"]], [dg["Rtune"]], "bo")
                ax.set_thetagrids(range(0, 360, 45), fontsize=fontsize - 6)
                ax.set_yticks([0.5, 1])
                ax.set_yticklabels([])
                ax.set_ylim([0, 1])
            else:
                missing(ax, "no DG")

            # --- Spatial STA (broad zoom) + ellipse overlay ---
            ax = fig.add_subplot(gs[row, 1])
            try:
                ellipse = sta_results[cell_nb]["center_analyse"]["EllipseCoor"]
                spatial = sta_results[cell_nb]["center_analyse"]["Spatial"]
                x0, y0 = ellipse[1], ellipse[2]
                utils.plot_sta(ax, spatial, ellipse)
                ax.set_xlim(x0 - rf_zoom, x0 + rf_zoom)
                ax.set_ylim(y0 + rf_zoom, y0 - rf_zoom)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                gaussian = utils.gaussian2D(spatial.shape, *ellipse)
                if ellipse[0] != 0:
                    ax_ellipses.contour(
                        np.abs(gaussian), levels=[0.6 * np.max(np.abs(gaussian))],
                        colors="k", linestyles="solid", alpha=0.8,
                    )
            except Exception as err:
                print(f"Warning: no spatial STA for cell {cell_nb} ({err}).")
                missing(ax, "no STA")

            # --- Temporal STA ---
            ax = fig.add_subplot(gs[row, 2])
            try:
                temporal = np.asarray(
                    sta_results[cell_nb]["center_analyse"]["Temporal"][-21:], dtype=float
                )
                ax.step(np.linspace(-21 / 30, 0, 21), temporal, "k", lw=2)
                ax.axhline(0, color="k", lw=0.5)
                ax.set_aspect(0.175)
                ax.axis("off")
                temporal_sum += temporal
                temporal_count += 1
            except Exception as err:
                print(f"Warning: no temporal STA for cell {cell_nb} ({err}).")
                missing(ax, "no STA")

            # --- Cell label ---
            ax = fig.add_subplot(gs[row, 3])
            ax.axis("off")
            ax.text(0, 0.5, f"Cell {cell_nb}", fontsize=fontsize, va="center")

            # --- Chirp PSTH ---
            ax = fig.add_subplot(gs[row, 4:8])
            try:
                psth = cell_data[cell_nb]["psth"]
                ax.plot(np.linspace(0, 32, len(psth)), psth)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.tick_params(labelsize=fontsize - 4)
                ax.locator_params(axis="y", nbins=3)
            except Exception as err:
                print(f"Warning: no chirp PSTH for cell {cell_nb} ({err}).")
                missing(ax, "no chirp PSTH")

        # --- Summary header (rows 0-1) ---

        # mean temporal STA
        ax = fig.add_subplot(gs[0, 3])
        if temporal_count:
            ax.plot(np.linspace(-21 / 30, 0, 21), temporal_sum / temporal_count, "k", lw=2)
            ax.set_aspect(0.175)
        ax.set_title("Mean temporal STA", fontsize=fontsize)
        ax.axis("off")

        ax_ellipses.set_title("RF ellipses", fontsize=fontsize)
        ax_ellipses.set_aspect("equal")
        ax_ellipses.set_xticks([])
        ax_ellipses.set_yticks([])

        # mean chirp PSTH
        ax = fig.add_subplot(gs[0, 4:8])
        ax.set_title("Mean chirp PSTH", fontsize=fontsize)
        try:
            rows = [selected_cells.index(c) for c in cluster_cells]
            ax.plot(np.linspace(0, 32, psth_z.shape[1]), np.mean(psth_z[rows, :], axis=0), "b")
        except Exception as err:
            print(f"Warning: mean chirp PSTH failed for cluster {icluster} ({err}).")
        ax.axis("off")

        # stimulus trace
        ax = fig.add_subplot(gs[1, 4:8])
        if euler_vec is not None:
            ax.plot(np.linspace(0, 32, 1600), euler_vec[151 : 151 + 1600, 1], color="k")
            ax.set_ylim([-100, 350])
            ax.set_yticks([])
            ax.set_xlabel("Time (s)", fontsize=fontsize - 2)
            ax.tick_params(labelsize=fontsize - 4)
            for name, spine in ax.spines.items():
                spine.set_visible(name == "bottom")
        else:
            missing(ax, "no stimulus trace")

        fig.savefig(os.path.join(fig_directory, f"Cluster_{icluster}.png"), dpi=200)
        plt.close(fig)
