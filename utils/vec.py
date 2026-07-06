import numpy as np
import os
import gc
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict



###########################################################
###########          Analysis from vec          ###########
###########################################################


def split_spikes_by_triggers(spike_train, triggers):
    """Return spikes between consecutive triggers.

    Parameters
    ----------
    spike_train : array-like
        Spike times in samples or seconds.
    triggers : array-like
        Trigger times in the same unit as ``spike_train``.
    """
    return [
        spike_train[(spike_train >= triggers[i]) & (spike_train < triggers[i + 1])]
        for i in range(len(triggers) - 1)
    ]


def group_triggers_by_sequence(triggers, vec):
    """Group triggers by sequence identifier.

    Parameters
    ----------
    triggers : array-like
        Trigger times.
    vec : array-like
        Sequence identifiers for each trigger.
    """
    sequences = defaultdict(list)

    keys = vec.astype(int).astype(str)
    for key, trigger in zip(keys, triggers):
        sequences[key].append(trigger)

    return dict(
        sequences
    )  # this dictionnary has its keys ordered as the vec. !! CAUTION !! works for python > 3.7 only


def get_spike_sequences(spike_times, trig_seq):
    """Align spike times to the beginning of each sequence.

    Returns a dictionary keyed by sequence identifier, with spike times
    referenced to the first trigger of each sequence.
    """
    trigs = [
        [trig_list[0], trig_list[-1] + np.mean(np.diff(np.array(trig_list)))]
        for trig_list in trig_seq.values()
    ]  # make a list of all first and last trig of each seq
    splited_spikes = [
        split_spikes_by_triggers(spike_times, seq_times)[0] for seq_times in trigs
    ]
    return dict(zip(trig_seq.keys(), splited_spikes))


def spike_sequences_to_raster(spikesequences, trig_seq, n_digit_for_rep=4):
    """Convert spike sequences into rasters grouped by sequence.

    Repetitions are stacked by sequence prefix.
    """

    rasters = defaultdict(
        list
    )  # more compliant than dict. Allows you to either use an existing key or create it with empty list and than use it if missing.

    for key in spikesequences.keys():
        rasters[key[:-n_digit_for_rep]].append(spikesequences[key] - trig_seq[key][0])
    return dict(rasters)


def spike_sequences_to_psth(raster, trig_seq, bin_size=0.025, n_digit_for_rep=4):
    """Compute a PSTH from rasterized spike sequences."""
    psth = {}
    for key in raster.keys():
        n_rep = len(raster[key])
        if key == "":
            seq_range = (
                0,
                trig_seq["0"][-1] - trig_seq["0"][0] + np.mean(np.diff(trig_seq["0"])),
            )
        else:
            rep_signature = "0" * (
                n_digit_for_rep
            )  # Create a string of zeros to pad the key
            seq_range = (
                0,
                trig_seq[key + rep_signature][-1]
                - trig_seq[key + rep_signature][0]
                + np.mean(np.diff(trig_seq[key + rep_signature])),
            )

        n_bin = int(seq_range[1] / bin_size)
        binned_spike_count = np.zeros((n_rep, n_bin))
        for i in range(n_rep):
            binned_spike_count[i, :] = np.histogram(
                raster[key][i], bins=n_bin, range=seq_range
            )[0]
        psth[key] = np.sum(binned_spike_count, axis=0) / n_rep

    return psth


def prompt_user_for_vec_file(vec_directory: str) -> tuple[int, str]:
    """
    Display the vec (stimulus) files available in a folder and ask the user to pick one.

    Args:
        vec_directory: Folder containing the .vec stimulus files.

    Returns:
        Selected vec number as integer.
        Selected vec file name as string.
    """
    available_vec = os.listdir(os.path.normpath(vec_directory))
    print("Which vec (stimulus) file describes this recording:")
    for num, vec_file in enumerate(available_vec):
        print(f"\t{num} --> {vec_file}")

    vec_number = int(input("Vec file number : "))
    vec_filename = available_vec[vec_number]
    print(f"Selected vec file: {vec_filename}\n")

    return vec_number, vec_filename


def find_vec_file(vec_filename: str, stim_directory: str) -> str:
    """Ask the user which stimulus file to use, from ``stim_directory``.

    ALWAYS prompts: it lists the files of the same type as ``vec_filename`` (e.g. all
    ``.vec`` files, or all ``.bin`` files) in ``stim_directory`` and asks which one to
    use. When ``vec_filename`` itself is present it is offered as the default, so you can
    just press Enter to accept it — but you can always pick a different one (e.g. your own
    version of the stimulus, with different parameters), which is why it never picks
    automatically.

    Args:
        vec_filename: the expected / default stimulus file name.
        stim_directory: folder holding the stimulus files (``params.stim_directory``).

    Returns:
        Full path to the chosen stimulus file.
    """
    if not os.path.isdir(stim_directory):
        raise FileNotFoundError(
            f"The stimulus folder does not exist:\n    {stim_directory}\n"
            "Set 'stim_directory' in params.py to the folder that holds your stimulus files."
        )
    ext = os.path.splitext(vec_filename)[1]  # ".vec" or ".bin"
    files = sorted(f for f in os.listdir(stim_directory) if f.endswith(ext))
    if not files:
        raise FileNotFoundError(
            f"No '{ext}' files found in:\n    {stim_directory}\n"
            "Copy the right file there, or set 'stim_directory' in params.py."
        )

    default_idx = files.index(vec_filename) if vec_filename in files else None
    print(f"\nStimulus files in {stim_directory}:")
    for i, f in enumerate(files):
        print(f"    {i} : {f}" + ("   <- default" if i == default_idx else ""))

    if default_idx is not None:
        answer = input(
            f"Pick a file number, or press Enter for the default ({vec_filename}): "
        ).strip()
        chosen = files[int(answer)] if answer else vec_filename
    else:
        print(f"(the expected file '{vec_filename}' is not in this folder)")
        answer = input("Pick the matching file number: ").strip()
        chosen = files[int(answer)]

    print(f"Using stimulus file: {chosen}\n")
    return os.path.join(stim_directory, chosen)


def build_spikes_per_sequence_dict(
    cells: list,
    spike_times: dict,
    stim_onsets: np.ndarray,
    vec_keys: np.ndarray,
    bin_size: float = 0.025,
    n_digit_for_rep: int = 4,
) -> tuple[dict, dict, dict]:
    """
    Split each cell's spikes into stimulus sequences and build a raster + PSTH per sequence.

    For every cell and every sequence type (a sequence key with its repetition
    digits removed) this stacks the repetitions into a raster, computes the PSTH,
    and stores the timing info needed to plot it.

    Args:
        cells: Cell/cluster identifiers to process.
        spike_times: Spike times in seconds per cell, as {cell_id: np.ndarray}.
        stim_onsets: Trigger times in seconds, one per row of the vec file.
        vec_keys: Sequence key of each trigger (the vec file's last column).
        bin_size: PSTH bin width in seconds.
        n_digit_for_rep: Number of trailing digits of a key that encode the
            repetition number; the remaining leading digits identify the sequence type.

    Returns:
        spikes_per_sequence_dict: {cell_id: {sequence_key: {"raster": [np.ndarray],
            "psth": np.ndarray, "triggers": {"start": float, "end": float,
            "rng": (0, duration_s)}}}}
        triggers_per_repetition: {sequence_key+rep: [trigger_times]} (ordered as the vec).
        spikes_per_repetition: {cell_id: {sequence_key+rep: spikes}} aligned to each rep.
    """
    spikes_per_sequence_dict = {}
    spikes_per_repetition = {}

    # {"<seq><rep>": [trigger times]} — one entry per repetition, ordered as the vec file.
    triggers_per_repetition = group_triggers_by_sequence(stim_onsets, vec_keys)

    rep_signature = "0" * n_digit_for_rep  # suffix of the first repetition, e.g. "0000"

    for cell in tqdm(cells):
        spikes_per_sequence_dict[cell] = {}

        # This cell's spikes, split per repetition: {"<seq><rep>": spikes from sequence start}.
        cell_spikes_per_rep = get_spike_sequences(
            spike_times[cell], triggers_per_repetition
        )

        # Stack repetitions of the same sequence type: {"<seq>": [spikes per repetition]}.
        raster = spike_sequences_to_raster(
            cell_spikes_per_rep, triggers_per_repetition, n_digit_for_rep=n_digit_for_rep
        )
        # Mean firing over repetitions, binned: {"<seq>": np.ndarray of spike counts}.
        psth = spike_sequences_to_psth(
            raster, triggers_per_repetition, bin_size=bin_size
        )

        spikes_per_repetition[cell] = cell_spikes_per_rep
        for sequence_key in raster.keys():
            if sequence_key == "":
                continue

            first_rep_triggers = triggers_per_repetition[sequence_key + rep_signature]
            duration_s = (
                first_rep_triggers[-1]
                - first_rep_triggers[0]
                + np.mean(np.diff(first_rep_triggers))
            )
            spikes_per_sequence_dict[cell][sequence_key] = {
                "raster": raster[sequence_key],
                "psth": psth[sequence_key],
                "triggers": {
                    "start": first_rep_triggers[0],
                    "end": first_rep_triggers[-1],
                    "rng": (0, duration_s),
                },
            }

    return spikes_per_sequence_dict, triggers_per_repetition, spikes_per_repetition


def plot_sequence(
    sequence: dict,
    ax_rast: plt.Axes,
    ax_psth: plt.Axes,
    color: str = "#B85A8F",
    smoothing: float = 0.4,
    fontsize: int = 18,
) -> None:
    """
    Draw the raster and PSTH of a single sequence (one cell, one sequence type).

    A "sequence" here is one entry of spikes_per_sequence_dict, i.e.
    spikes_per_sequence_dict[cell][sequence_key], holding its "raster", "psth" and
    "triggers" info (see build_spikes_per_sequence_dict). Pass your own axes so the
    plot can be customised or combined with others.

    Args:
        sequence: One spikes_per_sequence_dict entry with keys "raster", "psth", "triggers".
        ax_rast: Axis to draw the raster on (one row of spikes per repetition).
        ax_psth: Axis to draw the PSTH on (firing rate over time).
        color: Any matplotlib color, used for both plots.
        smoothing: PSTH smoothing strength between 0 (none) and 1 (very smooth).
        fontsize: Base font size for titles and axis labels (tick labels use fontsize - 2).
    """
    tick_fontsize = fontsize - 2

    # Raster: one line of spikes per repetition.
    ax_rast.eventplot(sequence["raster"], color=color)
    ax_rast.set_title("Raster plot", fontsize=fontsize)
    ax_rast.set_ylabel("N repetitions", fontsize=fontsize)

    # PSTH: turn the binned spike counts into a firing rate (spikes/s), then smooth it.
    time_range = sequence["triggers"]["rng"]
    counts = sequence["psth"]
    firing_rate = counts * (len(counts) / (time_range[1] - time_range[0]))
    firing_rate = smooth(firing_rate, smoothing)

    time_axis = np.linspace(time_range[0], time_range[1], len(firing_rate))
    ax_psth.fill_between(time_axis, firing_rate, 0, alpha=1, color=color)
    ax_psth.set_xlabel("Time (s)", fontsize=fontsize)
    ax_psth.set_ylabel("Firing rate (spikes/s)", fontsize=fontsize)
    ax_psth.set_ylim(bottom=-0.1, top=max(1, max(firing_rate)))

    # Bigger tick labels on both panels.
    for ax in (ax_rast, ax_psth):
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)


def save_sequence_figures(
    spikes_per_sequence_dict: dict,
    analysis_directory: str,
    color: str = "#B85A8F",
    smoothing: float = 0.4,
    clusters_as_folder: bool = True,
    fontsize: int = 18,
) -> None:
    """
    Plot and save a raster + PSTH figure for every cell x sequence-type pair.

    Figures already present on disk are skipped, so this is cheap to re-run.

    Args:
        spikes_per_sequence_dict: Output of build_spikes_per_sequence_dict.
        analysis_directory: Folder where the figures are saved.
        color: Plot color, passed to plot_sequence.
        smoothing: PSTH smoothing strength, passed to plot_sequence.
        clusters_as_folder: If True, make one folder per cell (a figure per sequence
            inside). If False, make one folder per sequence (a figure per cell inside).
        fontsize: Base font size, passed to plot_sequence (the figure title uses fontsize + 2).
    """
    if clusters_as_folder:
        dict_to_plot = spikes_per_sequence_dict
        element, scd_element = "Cell", "Sequence"
    else:
        dict_to_plot = reshape_dict(spikes_per_sequence_dict)  # swap cell/sequence nesting
        element, scd_element = "Sequence", "Cell"

    for elt in tqdm(dict_to_plot.keys()):
        item_directory = os.path.normpath(
            os.path.join(analysis_directory, f"{element}_{elt}")
        )
        os.makedirs(item_directory, exist_ok=True)

        for scd_elt in dict_to_plot[elt].keys():
            figure_path = os.path.join(item_directory, f"{scd_element}_{scd_elt}.png")
            if os.path.isfile(figure_path):
                continue  # already plotted, skip

            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
                figsize=(10, 10),
            )
            plot_sequence(
                dict_to_plot[elt][scd_elt],
                ax_rast=axs[0],
                ax_psth=axs[1],
                color=color,
                smoothing=smoothing,
                fontsize=fontsize,
            )
            plt.suptitle(f"{scd_element}_{scd_elt}", fontsize=fontsize + 2)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(figure_path)
            plt.close(fig)
            gc.collect()  # free memory between figures (many cells x sequences)


def smooth(
    scalars: list[float], weight: float
) -> list[float]:  # Weight between 0 and 1
    """
    Function to smooth a 1D numpy array before plotting
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def reshape_dict(original_dict):
    """
    This function allows you to reshape dictionnaries by reversing their keys.
    If you have {Cell1 : {key1: data, key2: data}, Cell2 : {key1: data, key2: data}}
    you will get {key1 : {Cell1: data, Cell2: data}, key2 : {Cell1: data, Cell2: data}}
    """
    reshaped_dict = {}

    for cell_number, seq_dict in original_dict.items():
        for seq_number, data_dict in seq_dict.items():
            if seq_number not in reshaped_dict:
                reshaped_dict[seq_number] = {}
            reshaped_dict[seq_number][cell_number] = data_dict

    return reshaped_dict


