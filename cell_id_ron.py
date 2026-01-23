import numpy as np
import pickle
import os
import glob
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
import params
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
import itertools
import time
from collections import defaultdict
from math import *

#Above is just everything from original utils.  Likely overkill but prevents annoying errors from not having a needed module.


import utils


def import_data_to_plot(params: dict):

    """ Load all data for plots

    Args:
        params (dict): experiment parameters as set by params.py

    Returns:
        #2026-01-21 RWD: add data types to all of these.  For now just listing what I see.

    

    Note: RWD: would suggest putting all these return variables into a plotting dict or object maybe something called  plotdata or something
    for now leaving as individual vars

    """

    #Variables----------------------------------------------

    #Experience (experiment) name
    exp = params.exp

    # Directory where the analysis is saved
    output_directory = params.output_directory

    # Output directory of phy
    phy_directory = params.phy_directory

    #Input-------------------------------------------------

    #Reads the checkerboard analysis directory
    check_directory = utils.find_Analysis_Directory(dir_type="Checkerboard")

    #Reads the DG analysis directory
    DG_directory = utils.find_Analysis_Directory(dir_type="DG")

    #Reads the CT analysis directory
    CT_directory = utils.find_Analysis_Directory(dir_type="CellTyping")

    #load chirp stimulus for plotting the profile
    vec_path = os.path.join('./ressources', r"Euler_50Hz_20reps_1024x768pix.vec")
    euler_vec = np.genfromtxt(vec_path)

    #load the check rasters of the recording of choice
    check_rast = np.load(os.path.join(check_directory, 'Check_rasters_data.npy' ),allow_pickle=True).item()

    #load the DG data
    DG_data = np.load(os.path.join(DG_directory,'DG_data_exp{}.pkl'.format(exp)),allow_pickle=True)
    
    #load the STA analysis results
    sta_results=np.load(os.path.join(check_directory,'sta_data_3D_fitted.pkl'),allow_pickle=True)
    cells=list(sta_results.keys())
    
    #load the chirp
    Chirp_data = np.load(os.path.join(CT_directory, '{}_cell_typing_data.pkl'.format(exp)), allow_pickle=True)

    
    

    return exp, output_directory, phy_directory, check_directory, DG_directory, CT_directory, euler_vec, check_rast, DG_data, sta_results, cells, Chirp_data

def select_and_save_good_cells(cells, cell_rpvs,output_directory):
        
    """ Select cells for furhter analysis given their refractory period violations percentage

    Args:
        cells
        cells_rpvs

    Returns:
        #2026-01-21 RWD: add data types to all of these.  For now just listing what I see.

        good_cells
    

    Note: 

    """
    good_cells=[]
    for cell_nb in cells:
        if cell_rpvs[cell_nb]['rpv']<0.5: good_cells.append(cell_nb)
            
    good_cells=np.array(good_cells)
    np.save(os.path.join(output_directory,r'Good_cells'), good_cells)
    print(good_cells)
    
    return good_cells

def create_id_cards_and_plots(cells,cell_rpvs, output_directory):
            
    """ creat and id card for each cell

    Args:

        cells
        output_directory

    Returns:
        #2026-01-21 RWD: add data types to all of these.  For now just listing what I see.

       Doesn't appear to have returns.  Saves ID_cards to output_directory as well as some figures
       RWD self-note: not clear if figures are the ID cards or ID cards and figures in this function are separate things
    

    Note: 

    """

    #Input----------------------------------------------------------

    fig_directory = os.path.normpath(os.path.join(output_directory,r'ID_cards'))
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    #Plotting--------------------------------------------------------

    for cell_nb in tqdm(cells[:]):
    
        # Create the figure
        fig = plt.figure(figsize=(10, 12))
    
        # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(7, 5,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.3, hspace=0.7)
    
    
        if Chirp_data[cell_nb]['type']=='Not assigned': cluster=''
        else: cluster = Chirp_data[cell_nb]['type']
            
        plt.suptitle("exp{} _c{}  - Cluster_group_{} ".format(exp,cell_nb,cluster), fontsize=20)
        #--------------------------------------------------
        # Plot the ISI
        ax = fig.add_subplot(gs[0:2, 0:1])
        ax.hist(cell_rpvs[cell_nb]['isi']*1000, bins = 100, range=(0,50))
        rpv = cell_rpvs[cell_nb]['rpv']
        nb_spikes= cell_rpvs[cell_nb]['nb_spikes']
        nb_rpv= cell_rpvs[cell_nb]['nb_rpv_spikes']
        ax.axvline(rpv_len, lw=0.5, color='k')
        ax.set_title("Interspike Interval histogram\n RPV = {}%. {}/{} spikes".format(round(rpv,4), int(nb_rpv), nb_spikes))
        ax.set_xlabel("Interspike time (ms)")
        ax.set_ylabel("Number of spikes")
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(0,color='k',lw=0.5)
    
    
        #--------------------------------------------------
        # Plot spatial STA
        ax = fig.add_subplot(gs[0:2, 3:5])
        ax.set_title("Spatial receptive field")
        spatial = sta_results[cell_nb]['center_analyse']['Spatial']
        spatial = spatial**2*np.sign(spatial)
        cmap='RdBu_r'
        im = ax.imshow(spatial, cmap=cmap,interpolation='gaussian')
        # plt.colorbar(im, ax=axs[1,2])
        abs_max = 0.5*max(np.max(spatial), abs(np.min(spatial)))
        im.set_clim(-abs_max,abs_max)
    
        #--------------------------------------------------
        # Plot checkerboard repeated sequence raster
        ax = fig.add_subplot(gs[4:6, 3:5])
        ax.eventplot(check_rast[cell_nb]["spike_trains"], color='k', alpha=1,linelengths=1)
        ax.set_title("Repeated white noise sequences")
        ax.set_xlabel("Time (s)")
        seq_lenght=check_rast[cell_nb]["repeated_sequences_times"][0][1]-check_rast[cell_nb]["repeated_sequences_times"][0][0]
        ax.set_xlim([0,seq_lenght])
        ax.set_ylim([0,None])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        #--------------------------------------------------
        # Plot checkerboard repeated sequence psth (superimposed)
        ax = fig.add_subplot(gs[6:7, 3:5])
        width = (check_rast[cell_nb]["repeated_sequences_times"][0][0]/int(1200/2))
        seq_lenght=check_rast[cell_nb]["repeated_sequences_times"][0][1]-check_rast[cell_nb]["repeated_sequences_times"][0][0]
        ax.bar(np.linspace(0,seq_lenght,int(1200/2))+width/2, check_rast[cell_nb]["psth"], width=1.3*width)
        ax.set_xlabel("Time (s)")
        ax.set_xlim([0,seq_lenght])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        #--------------------------------------------------
        # Plot temporal STA
        ax = fig.add_subplot(gs[0:2, 1:3])
        ax.set_title("Temporal receptive field")
        ax.step(np.linspace(-1,0,21), sta_results[cell_nb]['center_analyse']['Temporal'][-21:],color='k',lw=3)
        ax.set_xlabel("Time (s)")
        ax.axhline(0,color='k',lw=0.5)
        ax.set_yticks([])
        ax.axis('off')
    
        #--------------------------------------------------
        # Plot chirp psth
        ax = fig.add_subplot(gs[6:7, 0:3])
        ax.plot(np.linspace(0,32,800), Chirp_data[cell_nb]['psth'])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing rate (spikes/s)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 32)
    
        #--------------------------------------------------
        # Plot chirp raster
        ax = fig.add_subplot(gs[4:6, 0:3])
        ax.eventplot(Chirp_data[cell_nb]["spike_trains"], color='k', alpha=1)
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 20)
        ax.set_ylabel("#Trial")
        ax.set_title("Response to the chirp stimulus")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        #--------------------------------------------------
        # Plot chirp stimulus profile (superimposed)
        ax = fig.add_subplot(gs[3:4, 0:3])
        ax.plot(np.linspace(0,32,1600),euler_vec[0+151:151+1600,1], color='k',lw=0.75)
        # ax.set_ylim(-800,300)
        ax.set_yticks([])
    #     ax.set_title("Chirp stimulus profile")
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 32)
    
        #--------------------------------------------------
        # Plot Polar plot
        ax = fig.add_subplot(gs[2:4, 3:5],polar=True)
        try:
            atune=DG_data[cell_nb]['atune']
            R=DG_data[cell_nb]['Rtune']
            TuneSum=DG_data[cell_nb]['Tuning']
            IDX=DG_data[cell_nb]['IDX']
        
            theta = np.linspace(0, 2 * np.pi, 9)
            ax.plot([atune,atune],[0,R],'b-')
            ax.plot([atune],[R],'bo')
            ax.plot(theta, TuneSum)
            ax.fill(theta, TuneSum, 'b', alpha=0.1)
            
            ax.text(np.pi/2*6/8,2.6,'IDX = '+str(np.round(IDX,1)),size=18)
            ax.text(np.pi/2*6/9,2.2,'R = '+str(np.round(R,1)),size=18)
            
            ax.set_yticks([0,0.5,1,1.5,2])
            ax.set_yticklabels([0,'',1,'',2])
        
        
        
            #plot rasters of DG
            ax = fig.add_subplot(gs[2:3, 0:3])
            seq_sep=20
            seq_len=12
            ch_raster=DG_data[cell_nb]['rasters']
            ax.eventplot(ch_raster[:],color='k',lw=1,linelengths=0.95)
            for a in np.arange(8):
                ax.axvline(a*seq_sep,color='gray',lw=2)
                ax.axvline(a*seq_sep+seq_len,color='gray',lw=2)
                ax.axvline(a*seq_sep+seq_len/6,color='gray',ls='--',lw=1.5)
    
    
            ax.set_xlim([-seq_sep/2,seq_sep*8])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([6,26,46,66,86,106,126,146],[0,45,90,135,180,225,270,315])
            ax.set_title('DG rasters')
            
            fsave = os.path.join(fig_directory, 'Group{}_cell{}'.format(cluster,cell_nb) ) 
        
            fig.savefig(fsave+'.png',format='png',dpi=110)
            plt.close(fig)
    
        except:
            
            fsave = os.path.join(fig_directory, 'Group{}_cell{}'.format(cluster,cell_nb) ) 
    
            fig.savefig(fsave+'.png',format='png',dpi=110)
            plt.close(fig)  
    


    return
