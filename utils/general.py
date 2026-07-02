import numpy as np
import pickle
import os
import gc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
from scipy.signal import convolve
from skimage import measure
import itertools
import time
from collections import defaultdict
import matplotlib.path as mpltPath
from types import ModuleType

import params


# =====================================================================
# CLEAN UTILS (TO FINISH)
# =====================================================================


def plot_raster_and_psth(
    spike_trains: list,
    psth: list,
    ax_rast: plt.Axes,
    ax_psth: plt.Axes,
    seq_length: float,
    t0: float = 0,
    title: str = "Raster plot",
    fontsize: int = 12,
    fontsize_labels: int = None,
):
    """
    Plot a raster plot and PSTH for given data in given axes.

    Args:
        spike_trains: List of spike trains (one per repetition). Length should be equal to the number of repetitions of the sequence.
        psth: Peri-stimulus time histogram values (firing rate in spikes/s) for each time bin. Length should be equal to the number of time bins used to extract the sequence response.
        ax_rast: Matplotlib axis for the raster plot.
        ax_psth: Matplotlib axis for the PSTH.
        seq_length: Duration of the sequence in seconds.
        t0: Start time of the sequence (default 0).
        title: Title for the raster plot (default "Raster plot").
        fontsize: Font size for titles and labels (default 12).
        fontsize_labels: Font size for axis labels (default fontsize-2).

    Returns:
        bin_values: Time values corresponding to the center of each PSTH bin in seconds.
        bin_width: Width of each PSTH bin in seconds.
    """
    if fontsize_labels is None:
        fontsize_labels = fontsize - 2
    ax_rast.eventplot(spike_trains)
    ax_rast.set_title(title, fontsize=fontsize)
    ax_rast.set_ylabel("n repetition", fontsize=fontsize)

    nbins = len(psth)
    bin_edges = np.linspace(t0, t0 + seq_length, nbins)
    bin_width = np.diff(bin_edges)[0]
    bin_values = bin_edges + bin_width / 2

    ax_psth.bar(
        bin_values,
        psth,
        width=bin_width,
    )
    ax_psth.set_xlabel("Time (s)", fontsize=fontsize)
    ax_psth.set_ylabel("Firing rate (spikes/s)", fontsize=fontsize)
    for ax in [ax_rast, ax_psth]:
        ax.tick_params(axis="both", which="major", labelsize=fontsize_labels)
    return bin_values, bin_width


