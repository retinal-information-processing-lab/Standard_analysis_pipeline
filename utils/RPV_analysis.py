import numpy as np
import os



#############################################
######          ID card                ######
#############################################


def get_cell_rpvs(cells, phy_directory, rpv_len=2.0, fs=20000):
    spike_clusters = np.load(os.path.join(phy_directory, "spike_clusters.npy"))
    spike_times = np.load(os.path.join(phy_directory, "spike_times.npy"))

    cell_rpv = {}

    for cell_nb in cells:
        cell_rpv[cell_nb] = {}

        sp_times = (
            spike_times[np.where(spike_clusters == cell_nb)[0]] / fs
        )  # cell's sp_times in seconds

        interspike_intervals = compute_interspike_intervals(sp_times)  # np.diff

        # percentage of isi less than rpv_len milliseconds
        rpv = compute_refractory_period_violation(
            sp_times, duration=rpv_len, cell_nb=cell_nb
        )

        nb_rpv = compute_number_of_rpv_spikes(sp_times, duration=rpv_len)

        cell_rpv[cell_nb]["nb_spikes"] = len(sp_times)
        cell_rpv[cell_nb]["isi"] = interspike_intervals
        cell_rpv[cell_nb]["rpv"] = rpv
        cell_rpv[cell_nb]["nb_rpv_spikes"] = nb_rpv

    return cell_rpv


def compute_interspike_intervals(spike_times):
    return np.diff(spike_times).astype(np.float64)


def compute_number_of_rpv_spikes(spike_times, duration=2.0):
    isis = compute_interspike_intervals(spike_times)
    nb_rpv = np.count_nonzero(isis <= 1e-3 * duration)

    return float(nb_rpv)


def compute_refractory_period_violation(spike_times, duration=2.0, cell_nb=None):
    """
    spike_times : the spike times of the neuron to study
    duration : the duraiton of the refractory period, in ms
    """

    isis = compute_interspike_intervals(spike_times)
    nb_isis = len(isis)
    if nb_isis == 0:
        if cell_nb is None:
            print("This cell has no spikes")
        else:
            print("Cell {} has no spikes".format(cell_nb))
        return 0
    else:
        rpv = compute_number_of_rpv_spikes(spike_times, duration) / float(nb_isis) * 100
        return rpv


