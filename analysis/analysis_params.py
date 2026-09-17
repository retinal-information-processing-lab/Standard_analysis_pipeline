from pathlib import Path
data_dir = Path(r'C:\thijs\sono_data')
checkerboard_dir = Path(r'C:\thijs\sono_data\dmd_stimfiles')

recording_params = {
    'nb_channels': 256,
    'nb_bytes_by_datapoint': 2,
    'dtype':'uint16',
    'fs': 20000,
    'data_voltage_resolution': (2*4096) / (2**16),
}

checkerboard_params = {
    'nb_checks_x': 30,
    'nb_checks_y': 30,
    'stimulus_frequency': 30,
    'nb_frames_by_sequence': 1200,  # Number of frames in each checkerboard sequence
}

dmd_channel = 128
dmd_threshold = 1000
maximal_jitter = 0.25e-3,  # Maximal error admissible in sec for time gap between triggers
# nb_frames_by_sequence = 1200  # Number of frames in each checkerboard sequence


# Audreys params
# threshold  = 270e+3
# pxl_size_dmd = 3.5
# size_dmd = [760, 1020]