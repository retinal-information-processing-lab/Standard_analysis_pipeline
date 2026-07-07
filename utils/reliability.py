"""Response-reliability metrics for repeated stimulus sequences.

When a stimulus sequence is repeated many times, a reliable cell responds the same way
every time. The even/odd split-half correlation quantifies this: bin each repetition into
a PSTH, average the even-numbered and odd-numbered repetitions separately, and correlate
the two averages. The score is a Pearson correlation in [-1, 1] (~1 = highly reliable,
~0 = noise). It gives an objective sanity check and an objective way to discard unstable
portions of a recording (e.g. "below r = 0.3 we consider the response unusable").
"""

import numpy as np


def _bin_repetitions(raster, n_bins, time_range):
    """Bin each repetition's spike times into a PSTH. Returns a (n_reps, n_bins) array."""
    return np.array(
        [np.histogram(rep, bins=n_bins, range=time_range)[0] for rep in raster],
        dtype=float,
    )


def _even_odd_corr(rep_psths):
    """Pearson correlation between the mean of even-indexed and odd-indexed PSTH rows."""
    even, odd = rep_psths[0::2], rep_psths[1::2]
    if len(even) == 0 or len(odd) == 0:
        return np.nan
    mean_even, mean_odd = even.mean(0), odd.mean(0)
    if np.std(mean_even) == 0 or np.std(mean_odd) == 0:
        return np.nan  # a flat half -> correlation undefined
    return float(np.corrcoef(mean_even, mean_odd)[0, 1])


def even_odd_reliability(raster, n_bins, time_range):
    """Global even/odd split-half reliability of a repeated-sequence response.

    Args:
        raster: list of per-repetition spike-time arrays (one array per repetition),
            each referenced to the sequence onset — i.e. one ``sequence["raster"]`` from
            build_spikes_per_sequence_dict, or the ``spike_trains`` list of a raster.
        n_bins: number of PSTH bins to use (e.g. ``len(sequence["psth"])``).
        time_range: (start, end) of the sequence in seconds (e.g. ``(0, duration_s)``).

    Returns:
        Pearson correlation between the even- and odd-trial mean PSTHs, in [-1, 1]
        (~1 = highly reliable). ``np.nan`` if it cannot be computed (fewer than one even
        and one odd trial, or a flat half).
    """
    rep_psths = _bin_repetitions(raster, n_bins, time_range)
    return _even_odd_corr(rep_psths)


def running_reliability(raster, n_bins, time_range, window=10):
    """Even/odd reliability computed in a sliding window of consecutive repetitions.

    Repetitions are in presentation order (≈ recording time), so this reveals when a cell
    (or the retina) becomes unstable over the course of the recording. For each window of
    ``window`` consecutive repetitions the even/odd correlation is computed.

    Args:
        raster: list of per-repetition spike-time arrays (see even_odd_reliability).
        n_bins: number of PSTH bins to use.
        time_range: (start, end) of the sequence in seconds.
        window: number of consecutive repetitions per window (default 10).

    Returns:
        centers: repetition index at the centre of each window (float array).
        scores: even/odd reliability of each window (same length as ``centers``).
    """
    rep_psths = _bin_repetitions(raster, n_bins, time_range)
    n_reps = len(rep_psths)
    if n_reps < window:
        return np.array([]), np.array([])
    centers, scores = [], []
    for start in range(0, n_reps - window + 1):
        scores.append(_even_odd_corr(rep_psths[start : start + window]))
        centers.append(start + window / 2)
    return np.array(centers, dtype=float), np.array(scores, dtype=float)
