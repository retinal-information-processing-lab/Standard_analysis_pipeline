"""Module of functions for the drifting gratings analysis

contact: laquitainesteeve@gmail.com
"""

# import packages
import os
from matplotlib.pyplot import *
import numpy as np
import itertools
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

# import custom packages
import utils

def load_trigger_times_and_load_spikes(params: dict):
    """Load trigger times and load spikes

    Args:
        params (dict): experiment parameters
    
    Returns:
        cells (list[np.uint32]): TODO! ??
        spike_times (list[np.array]): TODO! ??
        stim_onsets (np.array): TODO! ??
        trigsinrep (int): TODO! ??
        seq_sep (int): TODO! ??
        seq_len (int): TODO! ??
        DG_directory (str): TODO! ??

    Note:
        Do not change values here unless debug/specific use!
        
        All the variables used in this part of the cell should always refer to your 'params.py' file
        unless you want to manually change them only for this run (i.e. debugging). 
        You may have to add those variable into the function you want to adapt as only the minimal 
        amount of var are currently given to functions as inputs.
    """
    # Experiment name
    exp = params.exp

    # select DG recording
    recording_names = params.recording_names
    output_directory = params.output_directory
    fs = params.fs

    # Input ------------------------------------

    # Input to select to recording number
    print(*['{} : {}'.format(i,recording_name) for i, recording_name in enumerate(recording_names)], sep="\n")
    recording_number = int(input("\nSelect DG recording : "))
    rec    = recording_names[recording_number]
    print("\nSelected file : {}".format(rec))

    DG_directory = os.path.normpath(os.path.join(output_directory,r'DG_Analysis_rec_{}'.format(recording_number)))
    if not os.path.isdir(DG_directory): os.makedirs(DG_directory)

    # select grating's speed
    # T = 2  # T=0 = fast, T=1 medium,  T=2 slow
    T = int(input("\nSelect Grating's speed (T=0 = fast, T=1 medium,  T=2 slow (default)) : "))
        
    # Processing ------------------------------------

    
    # Load triggers
    trig_data = utils.load_obj(os.path.normpath(os.path.join(params.triggers_directory,'{}_{}_triggers.pkl'.format(exp,rec))))
    
    # convert them in seconds
    stim_onsets = trig_data['indices']/fs  
    
    if T==0: 
        seq_len=3.96
        seq_sep=9
        trigsinrep = int(50 * seq_len)
        ttext = 'FAST'
    if T==1: 
        seq_len=6
        seq_sep=10
        trigsinrep = 50 * seq_len
        ttext = 'MEDIUM'
    if T==2: 
        seq_len=12
        seq_sep=20
        trigsinrep = 50 * seq_len
        ttext = 'SLOW'

    output_directory=params.output_directory
    spike_trains=utils.load_obj(os.path.join(output_directory, r'{}_fullexp_neurons_data.pkl'.format(exp)))

    # Load spikes of the selected recording
    cells=list(spike_trains.keys())
    spike_times=[]
    for cell in cells:
        spike_times.append(spike_trains[cell][rec])
    
    # report 
    print('\nTotal : {} neurons loaded \n\nClusters id :\n{}\n'.format(len(spike_trains.keys()),cells))
    print('--- Cell Done ---')
    return cells, spike_times, stim_onsets, trigsinrep, seq_sep, seq_len, DG_directory


def compute_dg_rasters(cells:list[np.uint32], 
                       spike_times: list[np.array], 
                       stim_onsets: np.array, 
                       trigsinrep: int, 
                       seq_len: int, 
                       seq_sep: int, 
                       DG_directory: str,
                       params: dict):
    """Compute the DG rasters

    Args:
        cells (list[np.uint32]): TODO! ??
        spike_times (list[np.array]): TODO! ??
        stim_onsets (np.array): TODO! ??
        trigsinrep (int): TODO! ??
        seq_len (int): TODO! ??
        seq_sep (int): TODO! ??
        DG_directory (str): TODO! ??
        params (dict): 
    
    Returns:
        saves data as a pickle file
    """
    # Processsing -----------------------------------------
    exp = params.exp

    # DG angle sequence order
    DG_seq = [0, 1, 2, 3, 4, 5, 6, 7, 4, 1, 5, 2, 0, 3, 7, 6, 1, 4, 0, 3, 2, 5, 6, 7, 5, 2, 3, 6, 1, 4, 7, 0]
    DG_seq = (np.ones(32)*7-DG_seq).astype('int')  #(angles go counterclockwise in the stim)
        
    n_angles=8
    n_repeats=4

    Test_set = {}
    Tune_data = {}
    DG_set={}
    Nspikes_set = {}  #a dict that per each cells has the tot nb of spikes that the total stimulus evoked


    i0=0
    iz=len(cells)

    for i in tqdm(np.arange(i0,iz), desc="Computing Direction Selectivity "):
        
        clus=cells[i]
        dg_sptimes = spike_times[i]
        #################################################
        base_fire = 0                                     #what is this?? How to calculate it??
        #################################################

        #--------------------
        # Get start times and make rasters
        nb_rep = len(stim_onsets)//trigsinrep    # nb_angles*n_repeats   (32)
        dg_rep_starts = []
        for n in np.arange(nb_rep):
            dg_rep_starts.append(stim_onsets[trigsinrep*n])  #the times at which each of the 32 gratings starts sweeping
        dg_count=np.zeros([n_angles],dtype='int')
        ch_raster=[]
        for rep in np.arange(4):
            ch_raster.append([])
        for n in np.arange(8,nb_rep):  #number between 8 and 32, why excluding first 8 gratings? (1 of the 4 repetitions)
            
            #given a grating, rep_sptimes are the times of the spikes it evoked
            if n == nb_rep-1:
                rep_sptimes = dg_sptimes[(dg_rep_starts[n]<dg_sptimes)&(dg_sptimes<dg_rep_starts[n]+seq_len)]
            else:
                rep_sptimes = dg_sptimes[(dg_rep_starts[n]<dg_sptimes)&(dg_sptimes<dg_rep_starts[n+1])]
                
            ch_raster[dg_count[DG_seq[n]]] = np.append(ch_raster[dg_count[DG_seq[n]]],rep_sptimes-dg_rep_starts[n]+DG_seq[n]*seq_sep)
            dg_count[DG_seq[n]]+=1
            #ch_raster is a list of 4 lists, one per repetition. Each list containes the times at which each orientation angle
            #evoked spikes. The single grating spike trains are artificially spaced by 20 seconds (seq_sep) for plotting purposes
        
        if not(list(itertools.chain(*ch_raster))):continue  # checking that ch_raster is not empty
        #--------------------------------------------------------------                    
        #--------------------------------------------------------------   
        
        Nspikes = len(ch_raster[0])+len(ch_raster[1])+len(ch_raster[2])+len(ch_raster[3])
        Nspikes_set.update({clus:Nspikes})

        TuneSum, atune,R,IDX, counts, maxcount, bins , DG_data = utils.compute_tuning(ch_raster,base_fire, seq_len, seq_sep)
        Tune_data.update({clus:[TuneSum,R,atune]})
        DG_set.update({clus:DG_data})
        #if Nspikes<10: continue

    """
        Saving
    """
        
    savef = os.path.join(DG_directory, 'DG_data_exp{}'.format(exp)) 
    utils.save_obj(DG_set, savef)    

    print('--- Cell Done ---')


def plot_dg_rasters(DG_directory, seq_sep, seq_len, params):
    """
        Input
    """
    exp = params.exp
    
    DG_set = utils.load_obj(os.path.join(DG_directory, 'DG_data_exp{}'.format(exp)))

    #folder where plots will be saved
    fig_directory = os.path.normpath(os.path.join(DG_directory,r'DG_figs'))
    if not os.path.isdir(fig_directory): os.makedirs(fig_directory)

    # to print plot in the notbook, set show=True
    show=False
    """
        Plotting
    """

    for cell in tqdm(DG_set.keys(), desc="Plotting "):
        ## Loading    
        #     ch_raster = DG_set[cell]['rasters']
        # DG_set[cell]['']
        #     DG_data = ({'IDX':IDX,'Tuning':TuneSum,'atune':atune,'Rtune':R, 'rasters': ch_raster, 'counts':counts, 'maxcount':maxcount, 'bins':bins})
            
        #--------plot the rasters-------------------
        fig = plt.figure(figsize=(12, 8))
        plt.suptitle("Cell {}".format(cell))
        
        gs = fig.add_gridspec(5, 8,
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.3, hspace=0.7)

        ax = fig.add_subplot(gs[0:2, 0:8])
        ax.eventplot(DG_set[cell]["rasters"][:],color='k',lw=1,linelengths=0.95)
        for a in np.arange(8):
            ax.axvline(a*seq_sep,color='gray',lw=2)
            ax.axvline(a*seq_sep + seq_len,color='gray',lw=2)
            ax.axvline(a*seq_sep + seq_len/6,color='gray',ls='--',lw=1.5)


        ax.set_xlim([-seq_sep/2, seq_sep*8])
        ax.set_ylim([-0.5,3.5+2+4+2])
        ax.set_yticks(np.arange(4))
        ax.set_ylabel('Repetition               Counts       ',size=10)
        ax.set_xlabel('Time (s) {8 angles}',size=10)
        #fig.suptitle(ttext+'    cluster '+str(clus) + '      '+'% spikes: ' +str(round(len(dg_sptimes)/len(sp_times)*100,1))+'    Nspikes '+str(Nspikes))
        plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
        ax.text(5,12,'0                      45                      90                    135                    180                   225                    270                   315')
        ax.axhline(3.5+2,color='k',lw=0.5)  # base_firing

        #--------------------------plot the histograms------------------------
        counts= DG_set[cell]['counts']/DG_set[cell]['maxcount']*4+3.5+2
        ax.hist(DG_set[cell]['bins'][:-1], DG_set[cell]['bins'],histtype='step',lw=1.5,color='darkblue',weights = counts)

        #--------------------------plot the polar plot left--------------

        ax = fig.add_subplot(gs[2:5, 1:4],polar=True)

        theta = np.linspace(0, 2 * np.pi, 9)
        # Arrange the grid into number of sales equal parts in degrees
        lines, labels = plt.thetagrids(range(0, 360, int(360/8)),np.arange(0,360,45))

        # Plot actual sales graph
        ax.plot(theta, DG_set[cell]['Tuning'])
        ax.fill(theta, DG_set[cell]['Tuning'], 'b', alpha=0.1)
    #         ax.plot(theta, TuneMax,'orange')

        ax.plot([DG_set[cell]['atune'],DG_set[cell]['atune']],[0,DG_set[cell]['Rtune']],'b-')
        ax.plot([DG_set[cell]['atune']],[DG_set[cell]['Rtune']],'bo')

        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([])
        ax.set_ylim([0,1])

        ax.text(np.pi*1/5,1.3,'IDX = '+str(np.round(DG_set[cell]['IDX'],1)),size=18)
        ax.text(np.pi*1/8,1.25,'R = '+str(np.round(DG_set[cell]['Rtune'],1)),size=18)

        #---------------------------plot the polar plot right (same as left but not limited between 0 and 1)------
        ax = fig.add_subplot(gs[2:5, 5:8], polar=True)

        ax.plot([DG_set[cell]['atune'],DG_set[cell]['atune']],[0,DG_set[cell]['Rtune']],'b-')
        ax.plot([DG_set[cell]['atune']],[DG_set[cell]['Rtune']],'bo')
        ax.plot(theta, DG_set[cell]['Tuning'])
        ax.fill(theta, DG_set[cell]['Tuning'], 'b', alpha=0.1)

        ax.set_yticks([0,0.5,1,1.5,2])
        ax.set_yticklabels([0,'',1,'',2])

        #-----------------------------------------------------------------------------------
        fsave = os.path.join(fig_directory, 'DG_resp_exp{}_Cell_{}'.format(exp, cell))
        if show:
            print(cell)
            plt.show(block=False)
        fig.savefig(fsave+'.png',format='png',dpi=90)
        close(fig)
        #--------------------------------------------------------------
        #--------------------------------------------------------------

    print('--- Cell Done ---')
    return fig