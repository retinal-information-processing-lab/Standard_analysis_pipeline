"""unit-tests

author: laquitainesteeve@gmail.com 

Usage:

    # run in git root repository, in the terminal
    pytest 

Requirements:

- 'data/20251219_PulsingGratings_PupilSize/Analysis/triggers/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'

"""
import os
import pytest
import utils_drifting_gratings as utils
import pickle 
import tempfile
import shutil

class TestLoadObj:
    """Test suite for load_obj function"""
    
    def test_load_obj_type(self):
        """test output type with Pulsing Gratings 
        known trigger data
        """
        # size (152K)
        test_data_path = 'tests/test_data/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'
        
        # load data
        output = utils.load_obj(test_data_path)

        # test
        assert isinstance(output, dict)


    def test_load_obj_keys(self):
        """test output keys with Pulsing Gratings 
        known trigger data
        """
        # size (152K)
        test_data_path = 'tests/test_data/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'

        # expected output keys
        expected_keys = ['indices', 'duration', 'trigger_type', 'indice_errors']
        
        # load data
        output = utils.load_obj(test_data_path)

        # test
        assert list(output.keys()) == expected_keys


class TestSaveObj:
    """Test suite for save_obj function"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup after test
        shutil.rmtree(temp_path)
    

    def test_save_simple_object(self, temp_dir):
        """Test saving a simple object"""
        obj = {"key": "value", "number": 42}
        filepath = os.path.join(temp_dir, "test_file.pkl")
        
        # call the function to test
        utils.save_obj(obj, filepath)
        
        # Verify file exists
        assert os.path.exists(filepath)
        
        # Verify content
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj

    def test_auto_add_pkl_extension(self, temp_dir):
        """Test that .pkl extension is added if missing"""
        obj = [1, 2, 3, 4, 5]
        filepath = os.path.join(temp_dir, "test_file")  # No extension
        
        # call function
        utils.save_obj(obj, filepath)
        
        # Check that .pkl was added
        expected_path = filepath + ".pkl"
        assert os.path.exists(expected_path)
        
        with open(expected_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj

    def test_dont_duplicate_pkl_extension(self, temp_dir):
        """Test that .pkl extension is not duplicated"""
        obj = "test string"
        filepath = os.path.join(temp_dir, "test_file.pkl")
        
        # call function
        utils.save_obj(obj, filepath)
        
        # should not create test_file.pkl.pkl
        assert os.path.exists(filepath)
        assert not os.path.exists(filepath + ".pkl")

    def test_create_nested_directories(self, temp_dir):
        """Test that nested directories are created"""
        obj = {"nested": True}
        filepath = os.path.join(temp_dir, "level1", "level2", "level3", "test.pkl")
        
        # call function
        utils.save_obj(obj, filepath)
        
        # test
        assert os.path.exists(filepath)
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj

    def test_save_various_object_types(self, temp_dir):
        """Test saving different Python object types"""
        test_objects = [
            42,  # int
            3.14,  # float
            "string",  # str
            [1, 2, 3],  # list
            {"a": 1, "b": 2},  # dict
            {1, 2, 3},  # set
            (1, 2, 3),  # tuple
            None,  # None
        ]
        
        for i, obj in enumerate(test_objects):
            filepath = os.path.join(temp_dir, f"test_{i}.pkl")

            # call function
            utils.save_obj(obj, filepath)
            
            with open(filepath, 'rb') as f:
                loaded_obj = pickle.load(f)
            assert loaded_obj == obj

    def test_overwrite_existing_file(self, temp_dir):
        """Test that existing files are overwritten"""
        filepath = os.path.join(temp_dir, "test.pkl")
        
        # Save first object
        obj1 = {"version": 1}

        # call function
        utils.save_obj(obj1, filepath)

        # Save second object (should overwrite)
        obj2 = {"version": 2}

        # call function
        utils.save_obj(obj2, filepath)
        
        # Verify second object was saved
        with open(filepath, 'rb') as f:
            loaded_obj = pickle.load(f)
        assert loaded_obj == obj2
        assert loaded_obj != obj1


class TestGetRecordingSpikes:
    """Test suite for get_recording_spikes function"""
    
    @pytest.fixture
    def sample_all_recs_spikes(self):
        """
        Create sample nested dictionary structure.
        Structure: {cell_nb: {recording_name: spike_data}}
        """
        return {
            "cell1": {
                "checkerboard": [0.1, 0.2, 0.3],
                "recording1": [1.1, 1.2, 1.3],
                "recording2": [2.1, 2.2, 2.3]
            },
            "cell2": {
                "checkerboard": [0.5, 0.6, 0.7],
                "recording1": [1.5, 1.6, 1.7],
                "recording2": [2.5, 2.6, 2.7]
            },
            "cell3": {
                "checkerboard": [0.9, 1.0, 1.1],
                "recording1": [1.9, 2.0, 2.1],
                "recording2": [2.9, 3.0, 3.1]
            }
        }
    
    def test_get_checkerboard_spikes(self, sample_all_recs_spikes):
        """Test retrieving checkerboard recording spikes"""
        result = utils.get_recording_spikes("checkerboard", sample_all_recs_spikes)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "cell1" in result
        assert "cell2" in result
        assert "cell3" in result
        assert result["cell1"] == [0.1, 0.2, 0.3]
        assert result["cell2"] == [0.5, 0.6, 0.7]
        assert result["cell3"] == [0.9, 1.0, 1.1]

    def test_get_recording1_spikes(self, sample_all_recs_spikes):
        """Test retrieving recording1 spikes"""
        result = utils.get_recording_spikes("recording1", sample_all_recs_spikes)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["cell1"] == [1.1, 1.2, 1.3]
        assert result["cell2"] == [1.5, 1.6, 1.7]
        assert result["cell3"] == [1.9, 2.0, 2.1]

    def test_nonexistent_recording_raises_keyerror(self, sample_all_recs_spikes):
        """Test that nonexistent recording raises KeyError"""
        with pytest.raises(KeyError):
            utils.get_recording_spikes("nonexistent", sample_all_recs_spikes)

    def test_empty_all_recs_spikes(self):
        """Test with empty dictionary"""
        all_recs_spikes = {}
        result = utils.get_recording_spikes("any_name", all_recs_spikes)
        
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_empty_spike_lists(self):
        """Test cells with empty spike lists"""
        all_recs_spikes = {
            "cell1": {
                "checkerboard": []
            },
            "cell2": {
                "checkerboard": [0.5, 0.6]
            }
        }
        
        result = utils.get_recording_spikes("checkerboard", all_recs_spikes)
        
        assert result["cell1"] == []
        assert result["cell2"] == [0.5, 0.6]