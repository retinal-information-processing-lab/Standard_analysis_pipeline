import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import AgglomerativeClustering
import scipy as sc

import utils

#############################################
######            Clustering           ######
#############################################


def _review_cells(cells, figure_path_for, prompt, crop=None):
    """Interactive quality review: show each cell's figure and keep it (y) or skip it (Enter/n).

    Args:
        cells: cell IDs to review.
        figure_path_for: function cell_id -> path of that cell's figure.
        prompt: the yes/no question, formatted with the cell id.
        crop: optional function image -> cropped image (e.g. to show only part of it).

    Returns:
        the list of kept cell IDs.
    """
    # Replace each figure in place (instead of stacking them) so there is no endless
    # scrolling, and show it large enough to read.
    from IPython.display import clear_output

    selected = []
    for i, cell_nb in enumerate(cells):
        fig_path = os.path.normpath(figure_path_for(cell_nb))
        if not os.path.isfile(fig_path):
            print(f"No figure for cell {cell_nb} ({fig_path}), skipping.")
            continue
        image = np.asarray(plt.imread(fig_path))
        if crop is not None:
            image = crop(image)
        clear_output(wait=True)  # remove the previous cell's figure + prompt
        print(f"Reviewing cell {i + 1}/{len(cells)}")
        plt.figure(r"Current cell", figsize=(15, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        if utils.is_yes(input(prompt.format(cell_nb))):
            selected.append(cell_nb)
        plt.close("all")
    return selected


def _selection_file(CT_directory, params):
    return os.path.normpath(
        os.path.join(CT_directory, f"{params.exp}_selected_cells_for_clustering.pkl")
    )


def load_cell_selection(CT_directory, params):
    """Return a previously saved clustering selection (dict), or None if there is none."""
    path = _selection_file(CT_directory, params)
    if os.path.isfile(path):
        return utils.load_obj(path)
    return None


def _save_selection(CT_directory, params, sta=None, chirp=None):
    """Persist the clustering selection, updating ONLY the part(s) given and keeping the
    rest. So saving the STA list never touches the chirp list and vice-versa. The combined
    ``selected_cells`` (good STA AND good chirp) is recomputed each time.

    Returns the saved dict.
    """
    data = load_cell_selection(CT_directory, params) or {
        "selected_cells_sta": [],
        "selected_cells_chirp": [],
        "selected_cells": [],
    }
    if sta is not None:
        data["selected_cells_sta"] = list(sta)
    if chirp is not None:
        data["selected_cells_chirp"] = list(chirp)
    data["selected_cells"] = [
        c for c in data["selected_cells_sta"] if c in data["selected_cells_chirp"]
    ]
    utils.save_obj(data, _selection_file(CT_directory, params))
    return data


def select_cells_by_sta(
    cells, check_directory, CT_directory, params, preselected=None, save_format="png"
):
    """Pick the cells with a well-defined STA, for clustering, and SAVE the selection.

    The STA list is chosen from, in order: ``preselected`` (if non-empty), a previously
    saved STA selection (reused so it is never lost), or an interactive review of each cell's
    checkerboard STA figure (y to keep, Enter/n to skip). The result is ALWAYS saved, updating only the
    STA part of the shared selection file — the chirp selection is preserved.

    To redo the review, pre-fill ``preselected`` or delete the selection file.

    Returns the list of STA-good cells.
    """
    saved = load_cell_selection(CT_directory, params)
    if preselected:
        sta = list(preselected)
    elif saved and saved.get("selected_cells_sta"):
        sta = saved["selected_cells_sta"]
        print(
            f"Reusing the saved STA selection ({len(sta)} cells). "
            "Pre-fill the list (or delete the selection file) to redo it."
        )
    else:
        sta = _review_cells(
            cells,
            lambda cid: os.path.join(
                check_directory, "Stas_figs", f"Cell_{cid}.{save_format}"
            ),
            "Cell {} — good STA?  [y = keep,  Enter/Esc/n = skip] : ",
        )
    _save_selection(CT_directory, params, sta=sta)
    return sta


def select_cells_by_chirp(
    cells, CT_directory, params, preselected=None, save_format="png"
):
    """Pick the cells with a nice chirp response, for clustering, and SAVE the selection.

    Same logic as select_cells_by_sta (preselected / reuse-saved / interactive), using the
    chirp raster figures — only the left ~2/3 (stimulus + raster + PSTH) is shown, the STA on
    the right is judged in the STA step. The result is saved, updating only the chirp part of
    the shared file — the STA selection is preserved — and the combined selection is
    recomputed.

    Returns the list of chirp-good cells.
    """
    saved = load_cell_selection(CT_directory, params)
    if preselected:
        chirp = list(preselected)
    elif saved and saved.get("selected_cells_chirp"):
        chirp = saved["selected_cells_chirp"]
        print(
            f"Reusing the saved chirp selection ({len(chirp)} cells). "
            "Pre-fill the list (or delete the selection file) to redo it."
        )
    else:
        chirp = _review_cells(
            cells,
            lambda cid: os.path.join(
                CT_directory,
                "Chirp_rasters+STA",
                f"{cid}_Chirp_raster+STA.{save_format}",
            ),
            "Cell {} — good chirp?  [y = keep,  Enter/Esc/n = skip] : ",
            crop=lambda image: image[:, : image.shape[1] * 2 // 3],
        )
    _save_selection(CT_directory, params, chirp=chirp)
    return chirp


def edit_cell_selection(
    CT_directory,
    params,
    sta_add=(),
    sta_remove=(),
    chirp_add=(),
    chirp_remove=(),
    remove_from_both=(),
):
    """Manually add / remove cells from the saved STA and chirp selections, non-destructively.

    Loads the CURRENT saved selection (the source of truth), applies the edits, and saves —
    the lists are never emptied or rebuilt from scratch, so nothing is lost if a list is left
    blank. ``remove_from_both`` removes a cell from both the STA and the chirp lists.

    Returns the updated ``(selected_cells_sta, selected_cells_chirp, selected_cells)``.
    """
    data = load_cell_selection(CT_directory, params) or {
        "selected_cells_sta": [],
        "selected_cells_chirp": [],
        "selected_cells": [],
    }

    def edited(current, add, remove):
        to_remove = set(remove) | set(remove_from_both)
        kept = [c for c in current if c not in to_remove]
        for c in add:
            if c not in kept:
                kept.append(c)
        return kept

    sta = edited(data["selected_cells_sta"], sta_add, sta_remove)
    chirp = edited(data["selected_cells_chirp"], chirp_add, chirp_remove)
    updated = _save_selection(CT_directory, params, sta=sta, chirp=chirp)
    print(
        f"STA: {len(updated['selected_cells_sta'])} cells | "
        f"chirp: {len(updated['selected_cells_chirp'])} cells | "
        f"selected for clustering (both): {len(updated['selected_cells'])}"
    )
    return (
        updated["selected_cells_sta"],
        updated["selected_cells_chirp"],
        updated["selected_cells"],
    )


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

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def restrict_array(array, value_min, value_max):
    array = array[array >= value_min]
    array = array[array <= value_max]
    return array.tolist()


def correlate_PersonPM(cell1, cell2, max_shift=25):
    assert max_shift < max(len(cell1), len(cell2))
    center = np.corrcoef(cell1, cell2)[0, 1]
    right = []
    left = []
    for t in range(1, max_shift + 1):
        right.append(np.corrcoef(cell1[t:], cell2[:-t])[0, 1])
        left.append(
            np.corrcoef(cell1[: -max_shift + t - 1], cell2[max_shift - t + 1 :])[0, 1]
        )
    return np.asarray(left + [center] + right)


#############################################
######      Cell typing (from chirp)   ######
#############################################


def select_dos_cells(
    selected_cells: list,
    dos_cells: list,
    DG_directory: str,
    CT_directory: str,
    params: dict,
):
    """Partition the clustering-selected cells into DOS and non-DOS.

    DOS = orientation- OR direction-selective (both kinds live in this group). They are
    split from the rest so the two groups can be cell-typed independently. Selection is
    either manual (pass a non-empty ``dos_cells`` list) or interactive: each cell's
    drifting-gratings figure (from the DG analysis, notebook 3) is shown and you confirm
    whether it is orientation- or direction-selective.

    Args:
        selected_cells (list): cells chosen for clustering (good STA + chirp).
        dos_cells (list): DOS cell IDs to use directly; if empty, select interactively
            (or reload a previously saved selection).
        DG_directory (str): the DG analysis directory (its ``DG_figs`` holds the figures).
            Get it with ``utils.find_analysis_directory(params.output_directory, "DG")``.
        CT_directory (str): cell typing output directory (the selection is saved here).
        params (dict): experiment parameters containing 'exp'.

    Returns:
        dos_cells (list): orientation-/direction-selective cells (subset of selected_cells).
        non_dos_cells (list): the remaining selected cells.
    """
    exp = params.exp
    dos_file = os.path.normpath(os.path.join(CT_directory, f"{exp}_dos_cells.pkl"))

    # Reuse a previously saved selection if none was given.
    if not dos_cells and os.path.isfile(dos_file):
        print(f"Loading previous DOS selection from : {dos_file}")
        dos_cells = utils.load_obj(dos_file)["dos_cells"]

    # Otherwise ask the user, showing each cell's DG figure. Each figure replaces the
    # previous one (no endless scrolling) and is shown large enough to read.
    if not dos_cells:
        from IPython.display import clear_output

        dg_fig_directory = os.path.normpath(os.path.join(DG_directory, "DG_figs"))
        print(
            "Selecting orientation-/direction-selective (DOS) cells from the DG plots ..."
        )
        dos_cells = []
        for i, cell_nb in enumerate(selected_cells):
            fig_path = os.path.join(
                dg_fig_directory, f"DG_resp_exp{exp}_Cell_{cell_nb}.png"
            )
            if not os.path.isfile(fig_path):
                print(f"No DG figure for cell {cell_nb}, skipping.")
                continue
            clear_output(wait=True)  # remove the previous cell's figure + prompt
            print(f"DOS selection — cell {i + 1}/{len(selected_cells)}")
            plt.figure("Current cell", figsize=(12, 11))
            plt.imshow(np.asarray(plt.imread(fig_path)))
            plt.axis("off")
            plt.show()
            if input(
                f"Cell {cell_nb} — orientation-/direction-selective (DOS)?  [y = yes,  Enter/Esc/n = no] : "
            ) in ["Y", "Yes", "y", "yes"]:
                dos_cells.append(cell_nb)
            plt.close("all")

    # Keep only DOS cells that are actually in the clustering set; the rest are non-DOS.
    dos_cells = [cell for cell in selected_cells if cell in dos_cells]
    non_dos_cells = [cell for cell in selected_cells if cell not in dos_cells]

    utils.save_obj(
        {"dos_cells": dos_cells, "non_dos_cells": non_dos_cells},
        os.path.join(CT_directory, f"{exp}_dos_cells"),
    )
    print(
        f"{len(dos_cells)} DOS (orientation/direction selective), {len(non_dos_cells)} non-DOS cells."
    )

    return dos_cells, non_dos_cells


def select_clusterable_cells(cell_data, sta_results, selected_cells):
    """Drop cells that cannot be clustered.

    A flat (silent) chirp PSTH or a flat/NaN STA temporal course both break the per-cell
    z-scoring used for clustering (which PCA rejects). Returns the list of clusterable
    cell ids and prints which ones were dropped.
    """
    valid_cells, dropped = [], []
    for cell_id in selected_cells:
        sta_tc = np.asarray(
            sta_results[cell_id]["sta_analysis"]["Temporal"][-21:], dtype=float
        )
        psth_ok = np.std(cell_data[cell_id]["psth"]) > 0
        sta_ok = np.all(np.isfinite(sta_tc)) and np.std(sta_tc) > 0
        (valid_cells if psth_ok and sta_ok else dropped).append(cell_id)
    if dropped:
        print(
            f"Dropping {len(dropped)} cell(s) with a flat/NaN PSTH or STA "
            f"(cannot be clustered): {dropped}"
        )
    return valid_cells


def compute_chirp_psth(cell_data, selected_cells):
    """Mean chirp PSTH per cell (averaged over the 20 repetitions).

    This is the chirp-response feature that goes into the clustering. Returns an array of
    shape (n_cells, n_time_bins).
    """
    n_rep = 20  # nb of repeats
    nt = 32  # total length (s)
    dt = 0.04  # bin size (s)
    time_bins = np.arange(0, nt + dt, dt)
    spikes = np.zeros((len(selected_cells), int(nt / dt), n_rep))
    for cell_index, cell_id in enumerate(selected_cells):
        spike_cell = cell_data[cell_id]["spike_trains"]
        for rep in range(n_rep):
            spikes[cell_index, :, rep] = np.histogram(spike_cell[rep], bins=time_bins)[
                0
            ]
    return np.mean(spikes, 2)


def compute_sta_time_course(sta_results, selected_cells):
    """Last 21 points of each cell's temporal STA — the STA feature used for clustering.

    Returns an array of shape (n_cells, 21).
    """
    sta_time_course = np.zeros((len(selected_cells), 21))
    for cell_index, cell_id in enumerate(selected_cells):
        sta_time_course[cell_index] = sta_results[cell_id]["sta_analysis"]["Temporal"][
            -21:
        ]
    return sta_time_course


def compute_ellipse_sizes(sta_results, selected_cells):
    """RF ellipse area (|pi * sigma_x * sigma_y|) per cell — the RF-size clustering feature.

    Returns the un-normalised sizes (array of length n_cells).
    """
    ell_size = np.zeros(len(selected_cells))
    for cell_index, cell_id in enumerate(selected_cells):
        width = sta_results[cell_id]["sta_analysis"]["EllipseCoor"][3]
        height = sta_results[cell_id]["sta_analysis"]["EllipseCoor"][4]
        ell_size[cell_index] = np.abs(np.pi * width * height)
    return ell_size


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
        os.path.join(check_directory, "sta_data_analysed_extended.pkl"),
        allow_pickle=True,
    )

    # Processing-----------------------------------------------------------

    # Keep only clusterable cells, then compute each per-cell clustering feature with its
    # own small metric function (defined above) so what feeds the clustering is explicit.
    selected_cells = select_clusterable_cells(cell_data, sta_results, selected_cells)
    n_cells = len(selected_cells)

    # Feature 1 -- chirp PSTH (z-scored) and its PCA.
    psth = compute_chirp_psth(cell_data, selected_cells)
    psth_z = sc.stats.zscore(psth, 1)
    if sparse:
        pca_transformer = SparsePCA(n_components_psth, random_state=0).fit(psth_z)
    else:
        # svd_solver="full" -> exact, deterministic SVD (the default "auto" picks the
        # randomized solver for this data shape, making the clustering vary run to run).
        pca_transformer = PCA(n_components_psth, svd_solver="full").fit(psth_z)
    psth_pca = pca_transformer.transform(psth_z)

    # Feature 2 -- STA temporal course (z-scored) and its PCA.
    STA_time_course = compute_sta_time_course(sta_results, selected_cells)
    sta_tc = sc.stats.zscore(STA_time_course[:, :], 1)
    if n_components_sta_tc > 0:
        pca_transformer2 = PCA(n_components_sta_tc, svd_solver="full").fit(sta_tc)
        sta_tc_pca = pca_transformer2.transform(sta_tc)

    # Assemble the clustering matrix: [PSTH PCs | STA PCs | normalised RF size].
    cluster_dataset = np.zeros((n_cells, n_components_psth + n_components_sta_tc + 1))
    cluster_dataset[:, :n_components_psth] = psth_pca
    if n_components_sta_tc > 0:
        cluster_dataset[
            :, n_components_psth : n_components_psth + n_components_sta_tc
        ] = sta_tc_pca

    # Feature 3 -- RF ellipse size, min-max normalised.
    ell_size = compute_ellipse_sizes(sta_results, selected_cells)
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
            os.path.join(
                utils.find_analysis_directory(params.output_directory, dir_type="DG"),
                f"DG_data_exp{exp}",
            )
        )
    except Exception as err:
        print(
            f"Warning: could not load DG tuning data ({err}); orientation plots skipped."
        )

    euler_vec = None
    try:
        vec_name = (
            "EulerStim180530_std.vec"
            if old
            else "Euler_50Hz_20reps_1024x768pix_std.vec"
        )
        euler_vec = np.genfromtxt(utils.find_vec_file(vec_name, params.stim_directory))
        if old:
            euler_vec = -euler_vec
    except Exception as err:
        print(
            f"Warning: could not load chirp stimulus vec ({err}); stimulus trace skipped."
        )

    def missing(ax, message):
        """Blank an axis and write a small 'missing' note in it."""
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            color="gray",
            style="italic",
        )

    cluster_ids = sorted(
        {
            cell_data[c]["type"]
            for c in cell_data
            if cell_data[c]["type"] != "Not assigned"
        }
    )

    for icluster in tqdm(cluster_ids, desc="Cluster summary figures"):
        cluster_cells = [
            c for c in selected_cells if cell_data[c].get("type") == icluster
        ]
        n_cells = len(cluster_cells)
        print(f"Cluster {icluster}: {n_cells} cells")

        fig = plt.figure(figsize=(16, (n_cells + 2) * 1.9), constrained_layout=True)
        gs = fig.add_gridspec(n_cells + 2, 8, width_ratios=[1, 1, 1, 0.6, 1, 1, 1, 1])
        fig.suptitle(
            f"Cell group {icluster} — {n_cells} cells",
            fontsize=fontsize + 4,
            fontweight="bold",
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
                ellipse = sta_results[cell_nb]["sta_analysis"]["EllipseCoor"]
                spatial = sta_results[cell_nb]["sta_analysis"]["Spatial"]
                x0, y0 = ellipse[1], ellipse[2]
                utils.plot_sta(ax, spatial, ellipse)
                ax.set_xlim(x0 - rf_zoom, x0 + rf_zoom)
                ax.set_ylim(y0 + rf_zoom, y0 - rf_zoom)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                # Scale bar (legend for the RF size), labelled with its length.
                if "Spatial_unit_size_um" in sta_results[cell_nb]["sta_analysis"]:
                    utils.add_scalebar(
                        ax,
                        scalebar_size_um=100,
                        pixel_size_um=sta_results[cell_nb]["sta_analysis"][
                            "Spatial_unit_size_um"
                        ],
                        scalebar_left_location=(0.95, 0.13),
                        scale_bar_color="black",
                        scale_bar_width=3,
                        fontsize=max(6, fontsize - 4),
                    )
                gaussian = utils.gaussian2D(spatial.shape, *ellipse)
                if ellipse[0] != 0:
                    ax_ellipses.contour(
                        np.abs(gaussian),
                        levels=[0.6 * np.max(np.abs(gaussian))],
                        colors="k",
                        linestyles="solid",
                        alpha=0.8,
                    )
            except Exception as err:
                print(f"Warning: no spatial STA for cell {cell_nb} ({err}).")
                missing(ax, "no STA")

            # --- Temporal STA ---
            ax = fig.add_subplot(gs[row, 2])
            try:
                temporal = np.asarray(
                    sta_results[cell_nb]["sta_analysis"]["Temporal"][-21:], dtype=float
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
            ax.plot(
                np.linspace(-21 / 30, 0, 21), temporal_sum / temporal_count, "k", lw=2
            )
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
            ax.plot(
                np.linspace(0, 32, psth_z.shape[1]),
                np.mean(psth_z[rows, :], axis=0),
                "b",
            )
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

        # Mark DOS (orientation- or direction-selective) clusters in the file name, e.g.
        # "Cluster_38_DOS.png". DOS and non-DOS cells are clustered separately, so a cluster
        # is DOS iff its cells carry the "dos" flag (set when the two labellings are merged).
        is_dos = any(cell_data[c].get("dos") for c in cluster_cells)
        cluster_name = f"Cluster_{icluster}_DOS" if is_dos else f"Cluster_{icluster}"
        fig.savefig(os.path.join(fig_directory, f"{cluster_name}.png"), dpi=200)
        plt.close(fig)


def plot_handmade_cluster(
    cluster_name,
    cell_list,
    cell_data,
    selected_cells,
    psth_z,
    sta_results,
    params,
    CT_directory,
    old: bool = False,
    fontsize: int = 16,
    rf_zoom: int = 10,
):
    """Build the cluster mosaic figure for a hand-picked group of cells.

    Same layout as ``create_cluster_summary_figure`` but for a single, user-defined
    cluster: give it a ``cluster_name`` and the ``cell_list`` you want to group, and the
    figure is saved as ``Cluster_<cluster_name>.png`` in ``CT_directory/Cell_typing/``.

    It works by re-labelling a temporary copy of ``cell_data`` (the originals are not
    touched) so that only ``cell_list`` forms the cluster, then reusing the standard
    figure builder.

    Note:
        The cells should be among the clustered cells (``selected_cells``); a cell not in
        that list is skipped, since its mean-chirp-PSTH row cannot be located in psth_z.
    """
    missing = [c for c in cell_list if c not in selected_cells]
    if missing:
        print(
            f"Note: {len(missing)} cell(s) are not in the clustered set and will be "
            f"skipped: {missing}"
        )

    temp_cell_data = {
        c: {
            **cell_data[c],
            "type": (cluster_name if c in cell_list else "Not assigned"),
        }
        for c in cell_data
    }
    create_cluster_summary_figure(
        temp_cell_data,
        selected_cells,
        psth_z,
        sta_results,
        params,
        CT_directory,
        old,
        fontsize,
        rf_zoom,
    )
    print(
        f"Saved 'Cluster_{cluster_name}.png' in "
        f"{os.path.join(CT_directory, 'Cell_typing')}"
    )
