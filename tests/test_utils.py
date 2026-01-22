"""unit-tests

author: laquitainesteeve@gmail.com 

Usage:

    # run in git root repository
    pytest 

Requirements:

- 'data/20251219_PulsingGratings_PupilSize/Analysis/triggers/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'

"""
import pytest
import utils_drifting_gratings as utils


def test_load_obj():
    """test load_obj function
    """
    # size (152K)
    test_data_path = 'data/20251219_PulsingGratings_PupilSize/Analysis/triggers/20251219_PulsingGratings_PupilSize_03_DG_50Hz_50%30ND_triggers.pkl'
    
    # load data
    output = utils.load_obj(test_data_path)

    # test
    assert isinstance(output, dict)