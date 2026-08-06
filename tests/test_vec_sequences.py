"""
Unit tests for the vec "sequence" analysis machinery (utils.vec).

These are the deterministic, dataset-free core of the pipeline: given trigger
times, a vec key column, and spike times, they slice spikes into per-repetition
rasters and PSTHs. This is exactly where the original drifting-gratings bug lived
(a repetition picking up too many spikes), so a regression test here is worth a lot.

Everything below is built from tiny SYNTHETIC inputs — no recording needed.

Run just these tests (no pytest required):
    python -m unittest tests.test_vec_sequences

Reminder on the key format: each vec key is  <sequence-type digits><repetition digits>,
where the last ``n_digit_for_rep`` digits (default 4) are the repetition number.
So "10001" = sequence-type "1", repetition "0001".
"""

import unittest

import numpy as np

import utils


class TestSplitSpikesByTriggers(unittest.TestCase):
    """The primitive: spikes falling between consecutive triggers, half-open [t_i, t_{i+1})."""

    def test_n_triggers_give_n_minus_1_bins(self):
        spikes = np.array([0.5, 1.5, 2.5])
        triggers = [0.0, 1.0, 2.0, 3.0]  # 4 triggers -> 3 bins
        bins = utils.split_spikes_by_triggers(spikes, triggers)
        self.assertEqual(len(bins), 3)

    def test_half_open_intervals(self):
        # Each spike lands in exactly one bin; bins are [t_i, t_{i+1}).
        spikes = np.array([0.5, 1.5, 2.5])
        triggers = [0.0, 1.0, 2.0, 3.0]
        bins = utils.split_spikes_by_triggers(spikes, triggers)
        np.testing.assert_array_equal(bins[0], [0.5])
        np.testing.assert_array_equal(bins[1], [1.5])
        np.testing.assert_array_equal(bins[2], [2.5])

    def test_spike_on_trigger_boundary_goes_to_later_bin(self):
        # A spike exactly on a trigger belongs to the bin that STARTS at that trigger,
        # never the previous one (>= start, < end). This is the anti-"over-collect" rule.
        spikes = np.array([1.0])
        triggers = [0.0, 1.0, 2.0]
        bins = utils.split_spikes_by_triggers(spikes, triggers)
        np.testing.assert_array_equal(bins[0], [])  # [0,1) excludes 1.0
        np.testing.assert_array_equal(bins[1], [1.0])  # [1,2) includes 1.0


class TestVecSequenceMachinery(unittest.TestCase):
    """
    Synthetic stimulus: 2 sequence-types x 2 repetitions x 3 triggers each,
    at 1-second spacing. Triggers 0..11.

        key "10000" -> triggers [0,1,2]   (type 1, rep 0)
        key "10001" -> triggers [3,4,5]   (type 1, rep 1)
        key "20000" -> triggers [6,7,8]   (type 2, rep 0)
        key "20001" -> triggers [9,10,11] (type 2, rep 1)
    """

    def setUp(self):
        self.triggers = np.arange(12, dtype=float)
        self.vec = np.array(
            [10000] * 3 + [10001] * 3 + [20000] * 3 + [20001] * 3, dtype=float
        )
        self.trig_seq = utils.group_triggers_by_sequence(self.triggers, self.vec)

    def test_group_triggers_by_sequence(self):
        self.assertEqual(
            list(self.trig_seq.keys()), ["10000", "10001", "20000", "20001"]
        )
        self.assertEqual(self.trig_seq["10000"], [0, 1, 2])
        self.assertEqual(self.trig_seq["20001"], [9, 10, 11])

    def test_group_preserves_vec_order(self):
        # Keys must come out in the order they appear in the vec (contiguous, rep 0 first).
        # Build a vec whose keys are intentionally out of numeric order and check
        # the dict preserves *insertion* (vec) order, not sorted order.
        triggers = np.arange(6, dtype=float)
        vec = np.array([20000] * 3 + [10000] * 3, dtype=float)
        trig_seq = utils.group_triggers_by_sequence(triggers, vec)
        self.assertEqual(list(trig_seq.keys()), ["20000", "10000"])

    def test_get_spike_sequences_assigns_spikes_to_the_right_sequence(self):
        spikes = np.array([0.5, 2.5, 3.5, 10.5])
        seq = utils.get_spike_sequences(spikes, self.trig_seq)
        np.testing.assert_array_equal(seq["10000"], [0.5, 2.5])
        np.testing.assert_array_equal(seq["10001"], [3.5])
        np.testing.assert_array_equal(seq["20000"], [])
        np.testing.assert_array_equal(seq["20001"], [10.5])

    def test_sequence_does_not_over_collect_into_the_next_one(self):
        # THE original-bug property: a spike just before the next sequence's first
        # trigger belongs to the current sequence; a spike AT that trigger does not.
        # Sequence "10000" spans triggers [0,1,2]; its window ends at 2 + mean_isi = 3.
        spikes = np.array([2.9, 3.0])
        seq = utils.get_spike_sequences(spikes, self.trig_seq)
        np.testing.assert_array_equal(seq["10000"], [2.9])  # 3.0 must NOT leak in
        np.testing.assert_array_equal(seq["10001"], [3.0])  # it belongs to the next seq

    def test_raster_groups_reps_by_type_and_aligns_each_to_zero(self):
        spikes = np.array([0.5, 2.5, 3.5, 10.5])
        seq = utils.get_spike_sequences(spikes, self.trig_seq)
        raster = utils.spike_sequences_to_raster(seq, self.trig_seq)
        # Prefixes are the keys with the 4 repetition digits stripped.
        self.assertEqual(set(raster.keys()), {"1", "2"})
        # Type "1": rep0 spikes [0.5,2.5] aligned to trigger 0; rep1 [3.5] aligned to trigger 3.
        self.assertEqual(len(raster["1"]), 2)
        np.testing.assert_array_almost_equal(raster["1"][0], [0.5, 2.5])
        np.testing.assert_array_almost_equal(raster["1"][1], [0.5])
        # Type "2": rep0 empty; rep1 [10.5] aligned to trigger 9.
        np.testing.assert_array_almost_equal(raster["2"][0], [])
        np.testing.assert_array_almost_equal(raster["2"][1], [1.5])

    def test_psth_averages_over_repetitions(self):
        spikes = np.array([0.5, 2.5, 3.5, 10.5])
        seq = utils.get_spike_sequences(spikes, self.trig_seq)
        raster = utils.spike_sequences_to_raster(seq, self.trig_seq)
        psth = utils.spike_sequences_to_psth(raster, self.trig_seq, bin_size=1.0)
        # Sequence window is 3 s -> 3 bins of 1 s. PSTH = mean spike count per bin over reps.
        # Type "1": rep0 -> [1,0,1], rep1 -> [1,0,0]; mean -> [1.0, 0.0, 0.5].
        np.testing.assert_array_almost_equal(psth["1"], [1.0, 0.0, 0.5])
        # Type "2": rep0 -> [0,0,0], rep1 -> [0,1,0]; mean -> [0.0, 0.5, 0.0].
        np.testing.assert_array_almost_equal(psth["2"], [0.0, 0.5, 0.0])


class TestNonDefaultRepDigits(unittest.TestCase):
    """
    Regression test for n_digit_for_rep != 4 through the whole build_spikes pipeline.

    A vec with 3-digit sequence types and 2-DIGIT repetitions (keys like "10000" =
    type "100", rep "00"), as produced by some rigs. build_spikes_per_sequence_dict
    threads n_digit_for_rep into the raster grouping but used to drop it on the PSTH
    call, which then padded the sequence key with 4 zeros ("100" + "0000") and raised
    KeyError "1000000". This checks the parameter is honoured end-to-end.
    """

    def setUp(self):
        # 2 sequence-types ("100", "200") x 2 reps x 3 triggers, 1 s apart. Triggers 0..11.
        # keys: 10000/10001 (type 100), 20000/20001 (type 200); reps are the last 2 digits.
        self.triggers = np.arange(12, dtype=float)
        self.vec_keys = np.array(
            [10000] * 3 + [10001] * 3 + [20000] * 3 + [20001] * 3, dtype=float
        )
        self.spikes = {7: np.array([0.5, 2.5, 3.5, 10.5])}

    def test_build_spikes_dict_honours_two_digit_reps(self):
        # Must NOT raise (the KeyError "1000000" regression) and must key by the
        # 3-digit sequence type, not the default-4-digit-stripped "1"/"2".
        result, _, _ = utils.build_spikes_per_sequence_dict(
            [7],
            self.spikes,
            self.triggers,
            self.vec_keys,
            bin_size=1.0,
            n_digit_for_rep=2,
        )
        self.assertEqual(sorted(result[7].keys()), ["100", "200"])
        # Same 2 reps x 3 s window as the default-digit test, just relabelled.
        self.assertEqual(len(result[7]["100"]["raster"]), 2)
        np.testing.assert_array_almost_equal(result[7]["100"]["psth"], [1.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(result[7]["200"]["psth"], [0.0, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
