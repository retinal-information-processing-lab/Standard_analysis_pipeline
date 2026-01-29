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
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA, SparsePCA
import scipy as sc
from sklearn.preprocessing import StandardScaler
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


def get_all_inputs_for_chirp_analysis(params: dict, old: bool):
    """Load trigger times and spike data, then select chirp stimulus type.

    NOTE : 'old' needs to be clarified

    Args:
        params (dict): Experiment parameters from params.py containing:
            - exp (str): Experiment name
            - recording_names (list): List of recording names
            - output_directory (str): Path to analysis output directory
            - triggers_directory (str): Path to trigger files
            - fs (float): Sampling rate of the MEA in Hz
        old (bool): If True, use old 2p room chirp. If False, use new 50Hz chirp.

    Returns:
        tuple: Contains:
            - cells (list[np.uint32]): List of neuron cluster IDs
            - spike_times (list[np.ndarray]): Spike times for each neuron
            - spike_trains (dict): Complete spike train data for all neurons
            - trig_data (dict): Trigger timing information
            - stim_onsets (np.ndarray): Stimulus onset times in seconds
            - check_directory (str): Path to checkerboard analysis directory
            - CT_directory (str): Path to cell typing output directory
            - old (bool): Chirp type flag (passed through)

    Note:
        Prompts user to select recording number from available recordings.
        Creates cell typing directory if it doesn't exist.
    """
    recording_names = params.recording_names
    output_directory = params.output_directory

    # Prompt user to select recording
    recording_number, rec = utils.prompt_user_for_recording(params, "chirp recording")
    print(f"\nSelected recording : {rec} \n")

    # Create cell typing directory
    CT_directory = utils.create_analysis_directory(
        params, recording_number, "CellTyping"
    )

    check_directory = utils.find_Analysis_Directory(dir_type="Checkerboard")

    # Load triggers
    stim_onsets, trig_data = utils.load_triggers(params, rec)

    # Load spike trains
    cells, spike_times = utils.load_spike_trains(params, rec)

    print(f"Total : {len(spike_times)} neurons loaded \n\nClusters id :\n{cells}\n")

    return (
        cells,
        spike_times,
        spike_times,
        trig_data,
        stim_onsets,
        check_directory,
        CT_directory,
        old,
    )


def compute_chirp_rasters(
    cells: list,
    spike_times: list,
    stim_onsets: np.ndarray,
    old: bool = False,
    n_bins: int = 800,
    n_bins_small: int = 16000,
):
    """Compute chirp stimulus responses including rasters, PSTH, and noise correlations.

    Args:
        cells (list): List of cell/cluster IDs
        spike_times (list[np.ndarray]): Spike times for each cell
        stim_onsets (np.ndarray): Stimulus onset times in seconds
        old (bool): If True, use old chirp parameters. Default False.
        n_bins (int): Number of bins for PSTH. Default 800.
        n_bins_small (int): Number of bins for noise analysis. Default 16000.

    Returns:
        dict: Nested dictionary with structure:
            cell_data[cell_id] = {
                'spike_times': original spike times,
                'repeated_sequences_times': list of [start, end] for each rep,
                'spike_trains': aligned spike trains for each rep,
                'psth': peri-stimulus time histogram (Hz),
                'mean_spikes_count_small_bin': mean spike count at fine timescale,
                'spikes_counts_small_bin': spike counts for each rep at fine timescale,
                'noise_small_bin': deviation from mean at fine timescale,
                'noise_large_bin': deviation from mean at coarse timescale
            }
    """

    # Processing-------------------------------------------
    print("Extracting cells responses to Chirp stimulus\n")

    if old:
        nb_repetitions = 30
        n_bins = 625  # Change here for old chirp n_bins
        rep_lenght = 25
    else:
        nb_repetitions = 20
        rep_lenght = 32
    time_bin = rep_lenght / n_bins  # in seconds

    cell_data = {}

    for idx, cell_nb in tqdm(enumerate(cells[:]), desc="Extraction"):
        if not cell_nb in cell_data.keys():
            cell_data[cell_nb] = {}

        # Get spike_times
        euler_sptimes = spike_times[idx]

        aligned_triggers = stim_onsets  # (in seconds)

        # Flashes: Get the repeated sequence times for the specified position
        repeated_sequences_times = []
        for i in range(0, nb_repetitions):
            if old:
                times = aligned_triggers[i * 999 : 999 * (i + 1)]
            else:
                times = aligned_triggers[i * 1600 + 151 : 151 + 1600 * (i + 1)]
            repeated_sequences_times += [[times[0], times[-1]]]

        # Build the spike trains corresponding to stimulus repetitions
        spike_trains = []
        for i in range(len(repeated_sequences_times)):
            spike_train = utils.restrict_array(
                euler_sptimes,
                repeated_sequences_times[i][0],
                repeated_sequences_times[i][1],
            )
            spike_trains += [spike_train]

        # Align the spike trains
        for i in range(len(spike_trains)):
            spike_trains[i] = spike_trains[i] - repeated_sequences_times[i][0]

        # Compute psth
        binned_spikes = np.empty((nb_repetitions, n_bins))  # 40 ms time bin
        spike_counts_small_bin = np.empty(
            (nb_repetitions, n_bins_small)
        )  # 2 ms time bin

        for i in range(nb_repetitions):
            binned_spikes[i, :] = np.histogram(
                spike_trains[i], bins=n_bins, range=(0, rep_lenght)
            )[0]
            spike_counts_small_bin[i, :] = np.histogram(
                spike_trains[i], bins=n_bins_small, range=(0, rep_lenght)
            )[0]

        psth = np.sum(binned_spikes, axis=0)
        mean_spikes_count = np.sum(spike_counts_small_bin, axis=0) / nb_repetitions

        # Transform spike count in firing rate
        binned_spikes = binned_spikes
        cell_data[cell_nb]["spike_times"] = spike_times
        cell_data[cell_nb]["repeated_sequences_times"] = repeated_sequences_times
        cell_data[cell_nb]["spike_trains"] = spike_trains
        cell_data[cell_nb]["psth"] = psth / time_bin

        cell_data[cell_nb]["mean_spikes_count_small_bin"] = mean_spikes_count
        cell_data[cell_nb]["spikes_counts_small_bin"] = spike_counts_small_bin
        cell_data[cell_nb]["noise_small_bin"] = np.subtract(
            spike_counts_small_bin, mean_spikes_count
        )
        cell_data[cell_nb]["noise_large_bin"] = np.subtract(
            binned_spikes, psth / rep_lenght
        )

    return cell_data


def plot_chirp_rasters(
    cells: list,
    cell_data: dict,
    CT_directory: str,
    check_directory: str,
    old: bool = False,
):
    """Generate and save chirp raster plots for all cells.

    Args:
        cells (list): List of cell/cluster IDs to plot
        cell_data (dict): Dictionary containing chirp response data for all cells
        CT_directory (str): Path to cell typing output directory
        check_directory (str): Path to checkerboard analysis directory containing STA results
        old (bool): If True, use old chirp parameters. Default False.

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
        n_reps = 30

    else:
        vec_path = os.path.join("./ressources", r"Euler_50Hz_20reps_1024x768pix.vec")
        euler_vec = np.genfromtxt(vec_path)
        rep_lenght = 32
        n_bins = 800
        n_reps = 20

    # Processing------------------------------------------------------------

    print(f"Saving Chirp raster plots in : {fig_directory} \n")

    time_bin = rep_lenght / n_bins  # in seconds
    for cell_nb in tqdm(cells[:]):

        fig = plt.figure(figsize=(19, 6))
        gs = GridSpec(
            8,
            19,
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.4,
            hspace=0,
            figure=fig,
        )
        #     fig=plt.figure(figsize=(16,6))

        # plot Chirp stimulus
        ax = fig.add_subplot(gs[0:2, :-3])
        if old:
            ax.plot(np.linspace(0, rep_lenght, 999), euler_vec[1:1000, 1], color="k")
        else:
            ax.plot(np.linspace(0, rep_lenght, 1600), euler_vec[151:1751, 1], color="k")
        ax.set_ylabel("Chirp Stimulus")
        ax.set_yticks([])
        ax.set_title("Cluster {}".format(cell_nb))
        ax.set_xlim([0, rep_lenght])

        # plot chirp raster
        ax = fig.add_subplot(gs[2:5, :-3])
        ax.eventplot(cell_data[cell_nb]["spike_trains"], color="k", lw=1, linelengths=1)
        ax.set_ylabel("#Trial")
        ax.set_xlim([0, rep_lenght])

        # plot chirp psth
        ax = fig.add_subplot(gs[5:, :-3])
        ax.step(np.linspace(0, rep_lenght, n_bins), cell_data[cell_nb]["psth"])
        ax.set_ylabel("Firing rate Hz \n ({} ms time bin)".format(int(time_bin * 1000)))
        ax.set_xlabel("Time (s)")
        ax.set_xlim([0, rep_lenght])

        # plot spatial STA
        ax = fig.add_subplot(gs[0:3, -3:])
        ax.set_title("STA", fontsize=12)
        spatial = sta_results[cell_nb]["center_analyse"]["Spatial"]
        spatial = spatial**2 * np.sign(spatial)
        cmap = "RdBu_r"
        image = ax.imshow(spatial, cmap=cmap, interpolation="gaussian")
        abs_max = 0.5 * max(np.max(spatial), abs(np.min(spatial)))
        image.set_clim(-abs_max, abs_max)
        ax.set_xticks([])
        ax.set_yticks([])

        fsave = os.path.join(fig_directory, "{}_Chirp_raster+STA".format(cell_nb))
        fig.savefig(fsave + ".png", format="png", dpi=90)
        plt.close(fig)

    print("--- Cell Done ---")

    return


def select_and_save_cells_for_clustering(
    cells, good_sta_cells: list, good_chirp_cells: list, CT_directory: str, params: dict
):
    """Update cell selection for clustering analysis.

    Args:
        good_sta_cells (list): Cell IDs with good STA quality (or empty list)
        good_chirp_cells (list): Cell IDs with good chirp responses (or empty list)
        CT_directory (str): Path to cell typing output directory
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
            CT_directory_path=CT_directory,
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
            - psth_z (np.ndarray): Z-scored PSTHs, shape (n_cells, n_time_bins)
            - sta_results (dict): Loaded STA analysis results
            - model: Fitted AgglomerativeClustering model with cluster labels

    Note:
        Displays diagnostic plots: PCA variance, dendrogram, cluster centroids.
        You want ~80% cumulative variance explained by PCA components.
        Adjust dist_thres to get approximately 50 clusters.
    """
    # Input---------------------------------------------------

    sta_results = np.load(
        os.path.join(check_directory, "sta_data_3D_fitted.pkl"), allow_pickle=True
    )

    # Processing-----------------------------------------------------------

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
    model = AgglomerativeClustering(
        distance_threshold=dist_thres, n_clusters=None
    )
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

    return psth_z, sta_results, model


def save_cluster_number_for_cells(selected_cells: list, cell_data: dict, model):
    """Save each cell's cluster assignment in the cell_data dictionary.

    Args:
        selected_cells (list): List of cell IDs that were included in clustering
        cell_data (dict): Dictionary containing data for all cells
        model: Fitted clustering model with labels_ attribute

    Returns:
        dict: Updated cell_data with 'type' field added for each cell.
              Clustered cells have integer cluster IDs, others have 'Not assigned'.

    Note:
        Run this after performing agglomerative clustering to save results.
        Modifies cell_data in place and also returns it.
    """

    # Save the cluster of each cell
    for cell_index in range(len(selected_cells)):
        cell_nb = selected_cells[cell_index]
        cell_data[cell_nb]["type"] = model.labels_[cell_index]
    cells = cell_data.keys()
    for cell in cells:
        if cell not in selected_cells:
            cell_data[cell]["type"] = "Not assigned"

    return cell_data


def compute_intracluster_crosscorr(
    cell_data: dict,
    sta_results: dict,
    selected_cells: list,
    n_bins: int,
    max_shift: int,
    old: bool,
):
    """Compute noise correlations within each cluster.

    Args:
        cell_data (dict): Dictionary containing spike data for all cells
        sta_results (dict): Dictionary containing STA spatial information
        selected_cells (list): List of cell IDs included in clustering
        n_bins (int): Number of time bins (not used, kept for compatibility)
        max_shift (int): Maximum time shift for cross-correlation analysis
        old (bool): If True, use old chirp parameters (30 reps). If False, use new (20 reps).

    Returns:
        dict: Updated cell_data with added fields for each cell:
            - 'corrs': List of time-shifted correlations with cluster members
            - 'mean_corr': Mean correlation across cluster members
            - 'distances': Spatial distances to cluster members
            - 'max_corr': Zero-lag correlations with cluster members

    Note:
        Only computes correlations for cells that were assigned to clusters.
        Cells marked as 'Not assigned' are skipped.
    """

    # Processing-----------------------------------

    if old:
        nb_repetitions = 30
        n_bins = 625  # Change here for old chirp n_bins
        rep_lenght = 25
    else:
        nb_repetitions = 20
        rep_lenght = 32
    time_bin = rep_lenght / n_bins  # in seconds

    # #--------------------
    # #Compute noise spike counts
    # for idx,cell_nb in tqdm(enumerate(cell_data.keys()), desc="Small time scall bining"):
    #     # Compute psth
    #     spike_counts_small_bin = np.empty((nb_repetitions,n_bins))   #2 ms time bin
    #     for i in range(nb_repetitions):
    #         spike_counts_small_bin[i,:] = np.histogram(cell_data[cell_nb]["spike_trains_small"][i], bins=n_bins, range=(0,rep_lenght))[0]
    #     # Take the mean spike counts over all repetitions
    #     mean_spikes_count = np.sum(spike_counts_small_bin, axis=0)/ nb_repetitions

    #     cell_data[cell_nb]["mean_spikes_count_small_bin"] = mean_spikes_count
    #     cell_data[cell_nb]["spikes_counts_small_bin"] = spike_counts_small_bin
    #     cell_data[cell_nb]["noise_small_bin"] = np.subtract(spike_counts_small_bin, mean_spikes_count)

    # --------------------
    # Compute correlation inside a cell type
    for icluster in tqdm(
        range(
            len(
                list(
                    set(
                        [
                            cell_data[cell]["type"]
                            for cell in cell_data.keys()
                            if cell_data[cell]["type"] != "Not assigned"
                        ]
                    )
                )
            )
        )[:],
        desc="Computing noise correlations within cell types",
    ):
        idx_cluster = sorted(
            list(
                np.where(
                    np.asarray(
                        [
                            cell_data[cell]["type"]
                            for cell in cell_data.keys()
                            if cell_data[cell]["type"] != "Not assigned"
                        ]
                    )
                    == icluster
                )[0]
            )
        )
        for index in idx_cluster:
            cell = selected_cells[index]
            cell1 = np.sum(cell_data[cell]["noise_small_bin"], axis=0) / nb_repetitions
            corrs = []
            dist = []
            max_corr = []
            for index_corr in [idx for idx in idx_cluster if idx != index]:
                cell_corr = selected_cells[index_corr]
                cell2 = (
                    np.sum(cell_data[cell_corr]["noise_small_bin"], axis=0)
                    / nb_repetitions
                )
                dist.append(
                    np.linalg.norm(
                        np.asarray(
                            sta_results[cell_corr]["center_analyse"]["EllipseCoor"][1:3]
                        )
                        - np.asarray(
                            sta_results[cell]["center_analyse"]["EllipseCoor"][1:3]
                        )
                    )
                )  # np.asarray is needed here due to a former bug in ellipses not fitted coordinates format
                corrs.append(
                    utils.correlate_PersonPM(cell2, cell1, max_shift=max_shift)
                )
                max_corr.append(
                    np.corrcoef(
                        np.sum(cell_data[cell]["noise_large_bin"], axis=0)
                        / nb_repetitions,
                        np.sum(cell_data[cell_corr]["noise_large_bin"], axis=0)
                        / nb_repetitions,
                    )[0, 1]
                )
            cell_data[cell]["corrs"] = corrs
            cell_data[cell]["mean_corr"] = np.mean(np.asarray(corrs), axis=0)
            cell_data[cell]["distances"] = dist
            cell_data[cell]["max_corr"] = max_corr

            return cell_data


def create_cluster_summary_figure(
    cell_data: dict,
    selected_cells: list,
    psth_z: np.ndarray,
    sta_results: dict,
    params: dict,
    CT_directory: str,
    old: bool,
):
    """Create comprehensive summary figures for each cluster showing all cells.

    Args:
        cell_data (dict): Dictionary containing chirp response data for all cells
        selected_cells (list): List of cell IDs included in clustering
        psth_z (np.ndarray): Z-scored PSTHs, shape (n_cells, n_time_bins)
        sta_results (dict): Dictionary containing STA analysis results
        params (dict): Experiment parameters containing 'exp' field
        CT_directory (str): Path to cell typing output directory
        old (bool): If True, use old chirp parameters. If False, use new chirp.

    Returns:
        None. Saves summary figures to CT_directory/Cell_typing/

    Note:
        Creates one figure per cluster showing:
        - Individual cell responses (orientation tuning, STA, chirp PSTH, correlations)
        - Cluster averages (mean STA, mean PSTH, RF positions)
        - Spatial correlation structure
    """

    # Input--------------------------------------------------

    # Name of the experiment
    exp = params.exp
    DG_set = utils.load_obj(
        os.path.join(utils.find_Analysis_Directory(dir_type="DG"), f"DG_data_exp{exp}")
    )

    fig_directory = os.path.normpath(os.path.join(CT_directory, r"Cell_typing"))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    if old:
        vec_path = os.path.join("./ressources", r"EulerStim180530.vec")
        euler_vec = -np.genfromtxt(vec_path)
    else:
        vec_path = os.path.join("./ressources", r"Euler_50Hz_20reps_1024x768pix.vec")
        euler_vec = np.genfromtxt(vec_path)

    # Plotting-----------------------------------------------------

    for icluster in tqdm(
        range(
            len(
                list(
                    set(
                        [
                            cell_data[cell]["type"]
                            for cell in cell_data.keys()
                            if cell_data[cell]["type"] != "Not assigned"
                        ]
                    )
                )
            )
        )[:]
    ):
        idx_cluster = sorted(
            list(
                np.where(
                    np.asarray(
                        [
                            cell_data[cell]["type"]
                            for cell in selected_cells
                            if cell_data[cell]["type"] != "Not assigned"
                        ]
                    )
                    == icluster
                )[0]
            )
        )
        print("Number of cells in cluster {}: {}".format(icluster, len(idx_cluster)))

        gs = GridSpec(len(idx_cluster) + 2, 10)

        if len(idx_cluster) < 7:
            yspan = 2
        else:
            yspan = -2
        fig = plt.figure(figsize=(22, (len(idx_cluster) + yspan) * 1.75))
        plt.suptitle("Cell group {}.\n {} cells.".format(icluster, len(idx_cluster)))

        # -------------------------------
        # Loop cells in cluster
        line = 2
        STAs = np.zeros(21)
        STAcount = 0
        waves = np.zeros(101)
        wavecount = 0
        ax0 = fig.add_subplot(gs[0:2, 1:3])
        ax_dist_corr = fig.add_subplot(gs[0:2, 8:])

        # Set the color cycle for the axis
        #     colors = ['blue', 'lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'steelblue', 'mediumslateblue', 'darkslateblue', 'midnightblue']
        #     ax_dist_corr.set_prop_cycle('color', colors)
        cum_dist = []
        cum_corr = []

        for index in sorted(idx_cluster):
            cell_nb = selected_cells[index]

            # -----------------
            # Plot temp STA
            ax = fig.add_subplot(gs[line, 2])

            #         ax.set_ylim([-4,4])
            ax.axis("off")
            ax.set_aspect(0.175)
            temporal_sta = sta_results[cell_nb]["center_analyse"]["Temporal"][-21:]
            ax.step(np.linspace(-21 / 30, 0, 21), temporal_sta, "k", lw=3)
            #         ax.set_title('Cluster {}'.format(cell_nb))
            ax.axhline(0, color="k", lw=0.5)

            STAs += temporal_sta
            STAcount += 1

            # -----------------
            # Plot temp STA avg
            ax = fig.add_subplot(gs[0, 3])

            ax.set_title("Temp STA")
            ax.set_ylim([-4, 4])
            ax.plot(np.linspace(-21 / 30, 0, 21), temporal_sta, lw=0.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("Time(s)")

            # -----------------
            # plot Chirp
            ax = fig.add_subplot(gs[line, 4:8])

            cell_index = selected_cells.index(cell_nb)
            ax.plot(np.linspace(0, 32, 800), cell_data[cell_nb]["psth"])
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            plt.locator_params(axis="y", nbins=3)

            if cell_data[cell_nb]["corrs"] != []:
                # -----------------
                # plot correlation over distance
                sorted_dists, sorted_max_corr = zip(
                    *sorted(
                        zip(
                            cell_data[cell_nb]["distances"],
                            cell_data[cell_nb]["max_corr"],
                        )
                    )
                )
                cum_dist += list(sorted_dists)
                cum_corr += list(sorted_max_corr)

                ax_dist_corr.scatter(
                    sorted_dists,
                    sorted_max_corr,
                    marker="o",
                    linewidths=1,
                    color="lightblue",
                    alpha=0.5,
                )
                ax_dist_corr.plot(
                    sorted_dists, sorted_max_corr, linestyle="-", linewidth=1, alpha=0.5
                )
                ax_dist_corr.set_title("Correlations")
                ax_dist_corr.set_zorder(20)
                ax_dist_corr.set_visible(True)
                # -----------------
                # plot correlations

                ax = fig.add_subplot(gs[line, 8:])
                plt.plot(
                    np.arange(
                        -int(len(cell_data[cell_nb]["mean_corr"]) / 2),
                        int(len(cell_data[cell_nb]["mean_corr"]) / 2) + 1,
                        1,
                    ),
                    cell_data[cell_nb]["mean_corr"],
                    linewidth=1,
                )
                plt.fill_between(
                    np.arange(
                        -int(len(cell_data[cell_nb]["mean_corr"]) / 2),
                        int(len(cell_data[cell_nb]["mean_corr"]) / 2) + 1,
                        1,
                    ),
                    np.min(np.asarray(cell_data[cell_nb]["corrs"]), axis=0),
                    np.max(np.asarray(cell_data[cell_nb]["corrs"]), axis=0),
                    alpha=0.35,
                )
                ax.set_xticks([])
                #             ax.set_yticks([0,0.5,1])

                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            else:
                ax_dist_corr.set_visible(False)
                sorted_dists = []
                sorted_max_corr = []

            # -----------------
            # plot Spatial STA
            ax = fig.add_subplot(gs[line, 1])

            parameters = sta_results[cell_nb]["center_analyse"]["EllipseCoor"]
            x0 = parameters[1]
            y0 = parameters[2]

            ax = utils.plot_sta(
                ax, sta_results[cell_nb]["center_analyse"]["Spatial"], parameters
            )
            ax.set_xlim(x0 - 4, x0 + 4)
            ax.set_ylim(y0 + 4, y0 - 4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            gaussian = utils.gaussian2D(
                sta_results[cell_nb]["center_analyse"]["Spatial"].shape, *parameters
            )
            if parameters[0] != 0:
                ax0.contour(
                    np.abs(gaussian),
                    levels=[0.6 * np.max(np.abs(gaussian))],
                    colors="k",
                    linestyles="solid",
                    alpha=0.8,
                )

            ax = fig.add_subplot(gs[line, 3])
            ax.annotate("Cluster {}".format(cell_nb), (0, 0.5), (0, 0.5), fontsize=15)

            ax.axis("off")

            # ----------------
            # plot orientation selectivity
            ax = fig.add_subplot(gs[line, 0], polar=True)

            theta = np.linspace(0, 2 * np.pi, 9)
            # Arrange the grid into number of sales equal parts in degrees
            lines, labels = plt.thetagrids(
                range(0, 360, int(360 / 8)), np.arange(0, 360, 45)
            )
            # Plot actual sales graph
            ax.plot(theta, DG_set[cell_nb]["Tuning"])
            ax.fill(theta, DG_set[cell_nb]["Tuning"], "b", alpha=0.1)
            ax.plot(
                [DG_set[cell_nb]["atune"], DG_set[cell_nb]["atune"]],
                [0, DG_set[cell_nb]["Rtune"]],
                "b-",
            )  # 2026-01-23 RWD: changed third call of DG_set[cell] to DG_set[cell_nb] as getting error cell is not defined and all other calls have cell_nb
            ax.plot([DG_set[cell_nb]["atune"]], [DG_set[cell_nb]["Rtune"]], "bo")
            ax.set_yticks([0, 0.333, 0.666, 1])
            ax.set_yticklabels([])
            ax.set_xticklabels([0, "", "", 135, "", 225, "", ""])
            ax.set_ylim([0, 1])

            line += 1

        # -----------------
        # avg STA
        STAs = STAs / STAcount
        ax = fig.add_subplot(gs[0, 3])
        ax.plot(np.linspace(-21 / 30, 0, 21), STAs, "k", lw=2)
        #     ax.set_ylim([-4,4])
        ax.set_aspect(0.175)
        ax.axis("off")

        # -----------------
        # size ellipses

        ax0.set_title("Ellipses")
        #     ax0.set_xlim(4,20)
        #     ax0.set_ylim(20,4)
        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # -----------------
        # mean chirp psth
        ax = fig.add_subplot(gs[0, 4:8])
        ax.set_title("Chirp psth")

        ax.plot(np.linspace(0, 32, 800), np.mean(psth_z[idx_cluster, :], 0), "b")
        ax.axis("off")

        # -----------------
        # plot chirp stim
        ax = fig.add_subplot(gs[1, 4:8])

        ax.plot(
            np.linspace(0, 32, 1600),
            euler_vec[0 + 151 : 151 + 1600, 1] * 1.0,
            color="k",
        )
        ax.set_yticks([])
        ax.set_ylim([-100, 350])
        ax.set_xlabel("Time(s)")
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # -----------------
        # plot mean correlation over distance

        if cell_data[cell_nb]["corrs"] != []:
            sorted_dists, sorted_max_corr = zip(*sorted(zip(cum_dist, cum_corr)))
            #     n_bin = int(len(sorted_dists)/1)
            #     print(n_bin)
            n_bin = 10

            xlim = np.linspace(min(sorted_dists), max(sorted_dists), n_bin + 1)
            mean = [
                np.mean(
                    np.asarray(sorted_max_corr)[
                        np.where(
                            np.logical_and(
                                sorted_dists >= xlim[i], sorted_dists <= xlim[i + 1]
                            )
                        )[0]
                    ]
                )
                for i in range(len(xlim) - 1)
            ]
            lmin = [
                min(
                    np.asarray(sorted_max_corr)[
                        np.where(
                            np.logical_and(
                                sorted_dists >= xlim[i], sorted_dists <= xlim[i + 1]
                            )
                        )[0]
                    ],
                    default=np.nan,
                )
                for i in range(len(xlim) - 1)
            ]
            lmax = [
                max(
                    np.asarray(sorted_max_corr)[
                        np.where(
                            np.logical_and(
                                sorted_dists >= xlim[i], sorted_dists <= xlim[i + 1]
                            )
                        )[0]
                    ],
                    default=np.nan,
                )
                for i in range(len(xlim) - 1)
            ]

            x_pos = np.linspace(min(sorted_dists), max(sorted_dists), n_bin)
            ax_dist_corr.plot(
                x_pos[np.isfinite(mean)],
                np.asarray(mean)[np.isfinite(mean)],
                color="#1f77b4",
            )
        #         ax_dist_corr.fill_between(x_pos[np.isfinite(lmin) & np.isfinite(lmax)],np.asarray(lmin)[np.isfinite(lmin) & np.isfinite(lmax)],np.asarray(lmax)[np.isfinite(lmin) & np.isfinite(lmax)], alpha=0.2)

        fsave = os.path.join(fig_directory, "Cluster_{}".format(icluster))
        fig.savefig(fsave + ".png", format="png", dpi=250)
        plt.close(fig)
