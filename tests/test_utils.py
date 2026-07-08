"""Unit tests for the directory / loader helpers in utils.loading.

These guard the leaf loaders whose signatures were made explicit (they now take
plain values instead of the params module). All inputs are synthetic (temp dirs
and mocks) — no recording needed.

Run (no pytest required):

    python -m unittest tests.test_utils
"""

import os
import shutil
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np

import utils


class TestCreateAnalysisDirectory(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after testing."""
        shutil.rmtree(self.test_dir)

    def test_creates_new_directory(self):
        """Test creating a new analysis directory."""
        result = utils.create_analysis_directory(self.test_dir, 1, "DG")

        self.assertTrue(os.path.isdir(result))
        self.assertIn("DG_Analysis_rec_1", result)

    def test_existing_directory(self):
        """Test that an existing directory is reused, not recreated."""
        result1 = utils.create_analysis_directory(self.test_dir, 2, "Checkerboard")
        result2 = utils.create_analysis_directory(self.test_dir, 2, "Checkerboard")

        self.assertEqual(result1, result2)
        self.assertTrue(os.path.isdir(result2))

    def test_different_analysis_types(self):
        """Test creating directories for different analysis types."""
        dg_dir = utils.create_analysis_directory(self.test_dir, 0, "DG")
        check_dir = utils.create_analysis_directory(self.test_dir, 0, "Checkerboard")

        self.assertIn("DG_Analysis_rec_0", dg_dir)
        self.assertIn("Checkerboard_Analysis_rec_0", check_dir)
        self.assertTrue(os.path.isdir(dg_dir))
        self.assertTrue(os.path.isdir(check_dir))


class TestFindAnalysisDirectory(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory with test folders."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after testing."""
        shutil.rmtree(self.test_dir)

    def test_single_directory_found(self):
        """Test finding a single matching directory."""
        os.makedirs(os.path.join(self.test_dir, "Checkerboard_Analysis_rec_0"))

        result = utils.find_analysis_directory(self.test_dir, "Checkerboard")

        self.assertIn("Checkerboard_Analysis_rec_0", result)

    @patch("builtins.input", return_value="1")
    @patch("sys.stdout", new_callable=StringIO)
    def test_multiple_directories_found(self, mock_stdout, mock_input):
        """Test selecting from multiple matching directories."""
        os.makedirs(os.path.join(self.test_dir, "DG_Analysis_rec_0"))
        os.makedirs(os.path.join(self.test_dir, "DG_Analysis_rec_1"))

        result = utils.find_analysis_directory(self.test_dir, "DG")

        self.assertIn("DG_Analysis_rec_1", result)

    def test_no_directory_found_raises_assertion(self):
        """Test that an assertion is raised when no matching directory exists."""
        with self.assertRaises(AssertionError):
            utils.find_analysis_directory(self.test_dir, "CellTyping")


class TestLoadStimOnsetFromTriggersPath(unittest.TestCase):
    def setUp(self):
        """Set up mock parameters."""
        self.params = MagicMock()
        self.params.fs = 20000  # Example sampling frequency

    @patch("utils.loading.load_obj")
    def test_basic_loading_and_conversion(self, mock_load_obj):
        """Test that triggers are loaded and converted to seconds correctly."""
        mock_load_obj.return_value = {
            "indices": np.array([0, 20000, 40000, 60000]),
            "trigger_type": "checkerboard",
        }

        stim_onsets = utils.load_stim_onset_from_triggers_path(
            "fake_path.pkl", self.params.fs, verbose=False
        )

        np.testing.assert_array_equal(stim_onsets, np.array([0.0, 1.0, 2.0, 3.0]))

    @patch("utils.loading.load_obj")
    def test_empty_triggers(self, mock_load_obj):
        """Test handling of empty trigger data."""
        mock_load_obj.return_value = {
            "indices": np.array([]),
            "trigger_type": "checkerboard",
        }

        stim_onsets = utils.load_stim_onset_from_triggers_path(
            "fake_path.pkl", self.params.fs, verbose=False
        )

        self.assertEqual(len(stim_onsets), 0)
        np.testing.assert_array_equal(stim_onsets, np.array([]))


if __name__ == "__main__":
    unittest.main()
