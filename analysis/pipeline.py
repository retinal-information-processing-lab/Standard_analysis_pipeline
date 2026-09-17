from matplotlib.backends.backend_nbagg import connection_info

from analysis.analysis_params import data_dir
from analysis.phy_data_loading import extract_phy_data
from analysis.analyse_checkerboard import (load_checkerboard_data,
                                           plot_all_rasters, checkerboard_params, plot_and_save_single_cell_rasters)
from utils import extract_from_sequence


def main():
    for d in data_dir.iterdir():
        if 'mouse' not in d.name and 'rat' not in d.name or not d.is_dir():
            continue
        sid = d.name

        if sid != '2026-09-09 mouse c57 758 Mekano6 C':
            continue


        output_directory = data_dir / sid / 'processed' / 'standard_analysis_pipeline'

        raw_dir = data_dir / sid / 'raw'
        phy_dir = data_dir / sid / 'processed' / 'sorted'
        if not raw_dir.exists():
            continue

        recording_names = [f.name for f in raw_dir.iterdir() if f.suffix == '.raw']

        # Creates ####_fullexp_neurons_data.pkl, ###_trigers.pkl and ###_triggers_data.pkl
        all_ok = extract_phy_data(raw_dir, phy_dir, output_directory, sid, overwrite=True)
        if not all_ok:
            print(f'\tproblem in phy data')
            continue

        # Do checkerboard analysis
        rec_name = [f for f in recording_names if 'checkerboard' in f]
        assert len(rec_name) == 1
        rec_name = rec_name[0]
        SWN = False  # Shifting white noise

        checkerboard_spikes, triggers, nb_repeats, cell_ids, _ = load_checkerboard_data(
                rec_name=rec_name,
                sid=sid,
                output_directory=output_directory,
            )

        repeated_sequence_portion = (
            0.5,
            1,
        )  # portion holding the repeated sequence (second half)
        non_repeated_sequence_portion = (
            0,
            0.5,
        )  # portion holding the random sequence (first half)

        res = dict()
        for cid in cell_ids:
            res[cid] = extract_from_sequence(
                cell_spikes=checkerboard_spikes[cid],
                triggers=triggers,
                nb_repeats=nb_repeats,
                stim_frequency=checkerboard_params['stimulus_frequency'],
                sequence_portion=repeated_sequence_portion,
                nb_frames_per_sequence=checkerboard_params['nb_frames_by_sequence'],
            )

        plot_all_rasters(
            rep_seq_data=res,
            cells_id=cell_ids,
            savename=data_dir / 'checkerboard_output' / f'{sid}.png',
        )

        # plot_and_save_single_cell_rasters(
        #     rep_seq_data=res,
        #     check_directory=output_directory / 'checkerboard',
        # )



if __name__ == '__main__':
    main()