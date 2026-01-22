"""Module of functions for the checkerboard analysis

contact: laquitainesteeve@gmail.com
"""

# import packages
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.gridspec import GridSpec

# import custom packages
import utils


# experimental design ---------------------------

def get_inputs(params: dict):
    """Get the parameters of the checkerboard experiment

    Args:
        params (dict): parameters of the experiments as 
        ... specified in params.py
    
    Returns:
        recording_number (TODO!): TODO!
        stimulus_frequency (TODO!): TODO!
        nb_checks_x (TODO!): TODO!
        check_directory (TODO!): TODO!

    Warning: 
        do not change values here unless debug/specific use!
    """
    # Getting the recording name
    print('Which of the following is the checkerboard: ')

    # Map the names of each recording to a number
    for num, rec in enumerate(params.recording_names):
        print(f'\t{num} --> {rec}')

    # ask the user to input the selected recording number
    recording_number = int(input('Checkerboard number : '))
    params.checkerboard_name = params.recording_names[recording_number]
    print('Selected recording: {}\n'.format(params.checkerboard_name))

    # ask the user to input the stimulus frequency
    stimulus_frequency = int(input("Select stimulus frequency (usually 30Hz) : "))

    # ask the user to input the number of checks used on the checkerboard
    nb_checks_x = int(input("Select number of checks on x (usually 40 for fine checkerboard) : "))
    nb_checks_y = int(input("Select number of checks on y (usually 40 for fine checkerboard) : "))
    
    # check that the checkerboard analysis path exists, else create it
    check_directory = os.path.normpath(os.path.join(params.output_directory,r'Checkerboard_Analysis_rec_{}'.format(recording_number)))
    if not os.path.isdir(check_directory): 
        os.makedirs(check_directory)
    return recording_number, stimulus_frequency, nb_checks_x, nb_checks_y, check_directory


def load_checkerboard_experiment_data(params: dict, check_directory:str, 
                                      nb_checks_x:int, nb_checks_y:int, 
                                      stimulus_frequency:int):
    """Process the checkerboard experiment data

    Args:
        params (dict): parameters of the experiments as 
        ... specified in params.py
        check_directory (str):
        nb_checks_x (int): 
        nb_checks_y (int):
        stimulus_frequency (int):

    Returns:
        checkerboard_spikes (TODO!): TODO!
        triggers (TODO!): TODO!
        nb_repeats (TODO!): TODO!
        cells_id (TODO!): TODO!
    """
    # load the checkerboard file where the triggers were recorded
    triggers_file = os.path.normpath(os.path.join(
        params.triggers_directory,r"{}_{}_triggers.pkl".format(
            params.exp, 
            params.checkerboard_name)
            ))
    triggers_data = utils.load_obj(triggers_file)

    # load the file where the spikes of all the recordings of 
    # the experiment are saved
    neurons_file  = os.path.normpath(os.path.join(
        params.output_directory,
        r'{}_fullexp_neurons_data.pkl'.format(params.exp)))
    all_recs_spikes = utils.load_obj(neurons_file)

    # keep only the spikes recorded during the checkerboard experiment
    checkerboard_spikes = utils.get_recording_spikes(params.checkerboard_name, all_recs_spikes)

    # get the ids of the cells that produced these spikes
    cells_id = list(checkerboard_spikes.keys())

    # get the triggers
    triggers = triggers_data['indices']/params.fs

    # get the number of repeats
    nb_repeats = int(len(triggers)/params.nb_frames_by_sequence)

    # get the duration of a sequence
    duration_sequence = int(params.nb_frames_by_sequence/stimulus_frequency)

    # report experiment parameters
    print("\nCheckerboard Stats :\n\t- {} min total duration \n\t- {} triggers \n\t- {} complete sequences\n\t- {} seconds per sequence\n".format(
        int(triggers_data['duration']/params.fs/60),len(triggers),
        nb_repeats, duration_sequence))

    # get the number of frames
    nb_frames = int((nb_repeats)*int(params.nb_frames_by_sequence/2))

    # get the stimulus path
    stimulus_path = os.path.normpath(os.path.join(
        check_directory, "checkerboard_{}x{}checks_{}frames.npy".format(nb_checks_x, 
                                                                        nb_checks_y, 
                                                                        nb_frames)))

    # check that the stimulus files exists, load it
    if os.path.isfile(stimulus_path):
        print("Stimulus file exists. Loaded from :\t {}".format(stimulus_path))
        checkerboard = np.load(stimulus_path)
    else:
        # else save
        print("Reconstructing the stimulus...")
        checkerboard = checkerboard_from_binary(
            nb_frames, nb_checks_x, nb_checks_y, 
            checkerboard_file=stimulus_path, 
            binary_source_path=params.binary_source_path
            )

    # report 
    print('Total : {} neurons loaded \n\nClusters id :\n{}\n'.format(len(checkerboard_spikes.keys()), cells_id))
    return checkerboard_spikes, triggers, nb_repeats, cells_id, checkerboard

# rasters and psths ---------------

def compute_rasters(checkerboard_spikes, triggers, nb_repeats, stimulus_frequency):

    # initialiser raster output
    raster_data = {}

    # report status
    print('Computing rasters...')

    # loop over the spikes recorded during the checkerboard experiment
    # get the rasters on repeated sequence
    for (cell_id, spike_times) in tqdm(checkerboard_spikes.items()):
        raster_data[cell_id] = utils.extract_from_sequence(spike_times, triggers, nb_repeats, stim_frequency = stimulus_frequency)
    return raster_data


def plot_rasters(raster_data, cells_id, ploting:bool=True):

    # Plot all the rasters. Takes a few seconds.
    if ploting:
        size = int(math.sqrt(len(cells_id)))+1

        # setup subplots
        fig, axs = plt.subplots(nrows = size, ncols=size, figsize = (50,50))
        print('Ploting...')
        for i in tqdm(range(size**2)):
            ax = axs[i//size,i%size]
            if i < len(cells_id):
                ax.eventplot(raster_data[cells_id[i]]["spike_trains"])
                ax.set(title = "Cell {}".format(cells_id[i]),xlabel='Time in sec', ylabel='N Repetitions')
            else : ax.set_visible(False)

        # format and close
        plt.tight_layout()
        plt.show(block=False)
        plt.close('all')


def save_plots(raster_data, cells_id, recording_number:int, 
               check_directory:str, params:dict):
    """Create a folder path with the saved raster and 
    psths plots, a file per cell.
    
    Args:
        raster_data
        cells_id
        recording_number (int): 
        check_directory (str):
        params (dict):

    Returns:
    """
    # report status
    print("Saving rasters ...")
    
    # figure path
    fig_directory = os.path.normpath(os.path.join(
        check_directory, r'Rasters_figs'.format(recording_number)))
    
    # ensure path figure exists
    if not os.path.isdir(fig_directory): 
        os.makedirs(fig_directory)
    
    # loop over cells
    for cell_nb in tqdm(cells_id):

        # setup subplots
        fig, axs = plt.subplots(nrows = 2,ncols = 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,10))

        # add title
        plt.suptitle(f'Cell {cell_nb}')
        
        # plot raster
        ax_rast = axs[0]
        ax_rast.eventplot(raster_data[cell_nb]["spike_trains"])
        ax_rast.set(title = "Raster plot", ylabel='N Repetitions')

        # plot firing rate psth
        ax_psth = axs[1]
        width = (raster_data[cell_nb]["repeated_sequences_times"][0][0]/int(params.nb_frames_by_sequence/2))
        seq_lenght = raster_data[cell_nb]["repeated_sequences_times"][0][1]-raster_data[cell_nb]["repeated_sequences_times"][0][0]
        ax_psth.bar(np.linspace(0, seq_lenght, int(params.nb_frames_by_sequence/2)) + width/2, 
                    raster_data[cell_nb]["psth"], width=1.3*width)
        ax_psth.set(xlabel='Time in sec', ylabel='Firing rate (spikes/s)')

        # format figure
        plt.subplots_adjust(wspace=0, hspace=0)

        # save figure
        fig_file = os.path.join(fig_directory,f'Cell_{cell_nb}.png')
        plt.savefig(fig_file, dpi=fig.dpi)

        # clear and close figure
        plt.clf()
        plt.close()
    
    # save raster data
    np.save(os.path.join(check_directory,'Check_rasters_data'), raster_data)


def plot_one_cell_raster_and_psth(raster_data, checkerboard_spikes, cells_id, params: dict):

    # report number of neurons
    print('Total : {} neurons found \n\nClusters id :\n{}\n'.format(len(checkerboard_spikes.keys()), cells_id))

    # ask the user to select a cell
    cell_nb = int(input("Select a cell: "))

    # setup plot
    fig, axs = plt.subplots(
        nrows = 2, ncols = 1, 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}, 
        figsize=(10, 10))

    # plot raster
    ax_rast = axs[0]
    ax_rast.eventplot(raster_data[cell_nb]["spike_trains"])
    ax_rast.set(title = "Raster plot", ylabel='N Repetitions')

    # plot psth
    ax_psth = axs[1]
    width = (raster_data[cell_nb]["repeated_sequences_times"][0][0]/int(params.nb_frames_by_sequence/2))
    seq_lenght = raster_data[cell_nb]["repeated_sequences_times"][0][1] - raster_data[cell_nb]["repeated_sequences_times"][0][0]
    ax_psth.bar(
        np.linspace(0,seq_lenght, int(params.nb_frames_by_sequence/2))+width/2, 
        raster_data[cell_nb]["psth"], width=1.3*width)
    ax_psth.set(xlabel='Time in sec', ylabel='Firing rate (spikes/s)')

    # format
    plt.suptitle(f'Cell {cell_nb}')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show(block=False)
    
    # close
    plt.close(fig)

# spike triggered averages ---------------

def compute_spike_triggered_average(checkerboard_spikes, checkerboard, 
                                    triggers, nb_repeats, stimulus_frequency, 
                                    check_directory: str,
                                    sequence_portion=(0, 0.5), 
                                    sta_data_filename = 'sta_data_3D.pkl'):

    # report status
    print('Computing STAs...')

    # initialize sta output
    sta_data = {}

    # loop over spikes of checkerboard experiment
    for (cell_id, spike_times) in tqdm(checkerboard_spikes.items()):
        
        # Get spikes on random sequences
        sta_data[cell_id] = utils.extract_from_sequence(spike_times, triggers, nb_repeats, stimulus_frequency, sequence_portion=sequence_portion)
        
        # Compute sta of the cell
        sta_3D = utils.compute_3D_sta(sta_data[cell_id], checkerboard, stimulus_frequency, cluster_id=cell_id) 
        
        # Adding data to the notebook dictionnary
        sta_data[cell_id]['sta_3D'] = sta_3D

    # save
    sta_data_file = os.path.normpath(os.path.join(check_directory, sta_data_filename))
    utils.save_obj(sta_data,sta_data_file)
    return sta_data_file, sta_data


def plot_one_cell_3D_spike_triggered_average(sta_data, checkerboard_spikes, cells_id):

    # report number of cells
    print('Total : {} neurons found \n\nClusters id :\n{}\n'.format(len(checkerboard_spikes.keys()), cells_id))
    
    # ask user to input a cell
    cell_id = int(input("Select a cell: "))

    # plot
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(8, 5, figure=fig)
    for i in range(40):
        ax = fig.add_subplot(gs[i//5,i%5])
        ax.imshow(sta_data[cell_id]['sta_3D'][i])
    plt.show(block=False)
    plt.close(fig)


def fit_ellipse_to_spike_triggered_average(
    checkerboard, 
    check_directory: str, 
    method: str="tom", 
    fitted_sta_filename='sta_data_3D_fitted.pkl'):

    """fit ellipse to STA
    
    Args:
        checkerboard (TODO!): TODO!
        check_directory (str): TODO!
        method (str): fitting method : 'tom', 'matias', 'gab' or 'basic'
    
    Returns:
        saves fitted STA data
    """
    # File loading the 3D sta dictionnary saved above in this notebook
    sta_data_file = os.path.normpath(os.path.join(check_directory, 'sta_data_3D.pkl'))
    sta_data = utils.load_obj(sta_data_file)

    # loop over cells
    for cell_id in tqdm(sta_data):

        # get 3D sta
        sta_3D = sta_data[cell_id]['sta_3D']
        
        if np.max(np.abs(sta_3D)) == 0:
            print(f'No ellipse fit on cell {cell_id}')
            sta_data[cell_id]["center_analyse"]   = {'Spatial':np.zeros(checkerboard[0].shape), 'Temporal':np.zeros(checkerboard.shape[0]), 'EllipseCoor':np.asarray([0, 0, 0, 0.001, 0.001, 0]), 'Cell_delay':np.nan}
            sta_data[cell_id]["surround_analyse"] = {'Spatial':np.zeros(checkerboard[0].shape), 'Temporal':np.zeros(checkerboard.shape[0]), 'EllipseCoor':np.asarray([0, 0, 0, 0.001, 0.001, 0]), 'Cell_delay':np.nan}
            sta_data[cell_id]["analyse_sta"]      = {'Spatial':np.zeros(checkerboard[0].shape), 'Temporal':np.zeros(checkerboard.shape[0]), 'EllipseCoor':np.asarray([0, 0, 0, 0.001, 0.001, 0]), 'Cell_delay':np.nan}

            continue
        
        # Perform the fitting for the current cell using one of the bellow method (default "analyse_sta_matias")
        if method=='tom':
            # New fitting of ellipse, more performant
            sta_data[cell_id]["center_analyse"] = utils.analyse_sta_tom(sta_3D, cell_id) 
        elif method=='matias':
            sta_data[cell_id]["center_analyse"] = utils.analyse_sta_matias(sta_3D, cell_id)
        elif method=='gab':
            sta_data[cell_id]["surround_analyse"] = utils.analyse_sta_gab(sta_3D, cell_id)
        elif method=='basic':
            sta_data[cell_id]["analyse_sta"]= utils.analyse_sta(sta_3D, cell_id)

    # ouput file of all fitted data
    fitted_file = os.path.normpath(os.path.join(check_directory, fitted_sta_filename))

    # save file
    utils.save_obj(sta_data, fitted_file)


def plot_sta_fitted_with_ellipse(cells_id, cells_to_plot:list, check_directory:str):
        
    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory,r'Stas_figs'))
    if not os.path.isdir(fig_directory): os.makedirs(fig_directory)

    # load sta
    sta_data = np.load(os.path.join(check_directory, 'sta_data_3D_fitted.pkl'), allow_pickle=True)

    # loop over all cells
    for cell_id in tqdm(cells_id[0:]):

        # get cell sta data
        sta = sta_data[cell_id]["center_analyse"]

        # setup subplots
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,7))

        # add title
        plt.suptitle(f'Cell {cell_id}')
        
        # plot spatial sta
        ax = axs[0]
        ax.imshow(sta['Spatial'])
        ax.set(title = "Spatial STA")

        # plot fitted ellipse
        ax = axs[1]
        ax.set(title = "Fitted Ellipse")

        try:
            ax = plot_sta(ax, sta['Spatial'], sta['EllipseCoor'])
        except:
            ax.imshow(sta_data[cell_id]["center_analyse"]['Spatial'])

        fig_file = os.path.join(fig_directory,f'Cell_{cell_id}.png')

        # save figure
        plt.savefig(fig_file, dpi=fig.dpi)

        # plot selected cells only
        if cell_id in cells_to_plot:
            plt.show()
        
        # close figure to free memory
        plt.close(fig)


def plot_sta_fitted_with_ellipse_by_tom(raster_data, cells_id, cells_to_plot:list, check_directory:str):

    # Folder where figure will be saved
    fig_directory = os.path.normpath(os.path.join(check_directory, r'Stas_figs'))
    if not os.path.isdir(fig_directory): os.makedirs(fig_directory)

    sta_data=np.load(os.path.join(check_directory,'sta_data_3D_fitted.pkl'),allow_pickle=True)

    # loop over cells
    for cell_id in tqdm(cells_id[0:]):

        # setup subplots
        fig, axs = plt.subplots(nrows = 1,ncols = 3, figsize=(30,10))

        plt.suptitle(f'Cell {cell_id}')
        sta=sta_data[cell_id]["center_analyse"]
        ax = axs[0]
        ax.eventplot(raster_data[cell_id]["spike_trains"])
        ax.set(title = "Raster plot", ylabel='N Repetitions')

        ax = axs[1]
        ax.set(title = "Fitted Ellipse")

        try:
            ax = plot_sta_tom(ax, sta['Spatial'],sta['EllipseCoor'])
        except: 
            ax.imshow(sta_data[cell_id]["center_analyse"]['Spatial'])

        ax=axs[2]
        ax.set(title = "Temporal STA")
        ax.plot(sta['Temporal'])
        ax.set_ylim([-1,1])

        fig_file = os.path.join(fig_directory,f'Cell_{cell_id}.png')
        plt.savefig(fig_file, dpi=fig.dpi)

        # plot selected cells only
        if cell_id in cells_to_plot:
            plt.show()

        # close figure to free memory
        plt.clf()
        plt.close(fig)