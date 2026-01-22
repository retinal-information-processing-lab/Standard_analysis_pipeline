"""unit-tests

author: laquitainesteeve@gmail.com 

Usage:

    # run in git root repository, in the terminal
    pytest 

Requirements:

- 'data/20251219_PulsingGratings_PupilSize/Analysis/triggers/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'

"""
import pytest
import utils_drifting_gratings as utils

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