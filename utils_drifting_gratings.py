import numpy as np
import pickle
import os
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

def save_obj(obj, name ):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open( os.path.normpath(name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
        Generic function to load a bin obj with pickle protocol

    Input :
        - name (str) : path to where the obj is
    Output :
        - (python object) : loaded object
        
    Possible mistakes :
        - Wrong path 
    """
    
    if os.path.dirname(os.path.normpath(name)) != '':
        os.makedirs(os.path.dirname(os.path.normpath(name)), exist_ok=True)
    else:
        name = os.path.join(os.getcwd(),os.path.normpath(name))
    if name[-4:]!='.pkl':
        name += '.pkl'
    with open(os.path.normpath(name), 'rb') as f:
        return pickle.load(f)


def compute_tuning(ch_raster,base_fire, seq_len, seq_sep, n_repeats=4):
    ###########################################################
    # computing tuning
    merged = list(itertools.chain(*ch_raster))   #all the spike times of all the 32 gratings. In this way when I bin I am
                                                 #binning per each of the 8 angles the responses to all the 4 repetitions of
                                                 #that angle

    nbins = 8*10*20                 #totoal nb of bins  (1600)
    binsize = seq_sep*8*1000//nbins #bin size in ms     (100)
    binsec = 1000//binsize          #nb bins per second  (10)
    base_fire=base_fire*(seq_sep*8/nbins)*n_repeats
    
    bins =  np.linspace(0,seq_sep*8,nbins+1)
    counts, bins = np.histogram(merged,bins=bins)   #binning the spike times of all the repetitions at once
    counts=counts-base_fire
    maxcount = np.amax(counts)

    #for plotting purposes, counts has 1600 bins, 10 each second of the 160 seconds. But some of this bins are fake because
    #the seq_sep (20 secs for the slow gratings) added in ch_raster is longer than the actual seq_len (12 secs for slow grating),
    #in which the stimulus was presented. So the last 8 secs after each angle have to have 80 empty.
    
        #--------------------------
    TuneSum = np.zeros(9)
    VxS=0
    VyS=0

    for a in np.arange(8):
        #################################################
        #per each angle I select the bins that go from 2 secs after the grating onset to the grating offset. Why?
        sel_bins = np.copy(counts[ int(seq_len*1000/6)//binsize + int(seq_sep*binsec*a): int(seq_len*binsec + seq_sep*binsec*a)])  
        #################################################

        TuneSum[a] = np.sum(sel_bins)    #per each angle these are all the spikes that the cell fired during the 4 repetitions
                                         #of that angle from 2 to 12 seconds
        #print(TuneSum[a])
        VxS+= np.cos(np.pi*a*45/180)*TuneSum[a]
        VyS+= np.sin(np.pi*a*45/180)*TuneSum[a]
#             VxM+= np.cos(np.pi*a/180)*TuneMax[a]
#             VyM+= np.sin(np.pi*a/180)*TuneMax[a]
        if a==0:
            TuneSum[a+8] = np.sum(sel_bins)

############################    
    if sum(TuneSum)==0:
        DG_data = ({'IDX':0,'Tuning':TuneSum,'atune':0,'Rtune':0, 'rasters': np.zeros( (4,len(bins)) ),'counts':counts, 'maxcount':maxcount, 'bins':bins})
        return np.zeros(9), 0, 0, 0, counts, maxcount, bins, DG_data
############################
    VxS=VxS/np.amax(TuneSum) 
    VyS=VyS/np.amax(TuneSum) 

    TuneSum=TuneSum/np.amax(TuneSum)
    atune = np.arctan2(VyS,VxS)
    R = np.sqrt(VyS**2+VxS**2)
   
    angle = int(np.round(atune/np.pi*4 ))
        
    IDX = (TuneSum[:-1][angle]-TuneSum[:-1][int((angle+4)%8)])/(TuneSum[:-1][angle]+ TuneSum[:-1][int((angle+4)%8)])
    if IDX<-0.2:
        angle2=angle+1
        IDX = (TuneSum[:-1][angle2]-TuneSum[:-1][int((angle2+4)%8)])/(TuneSum[:-1][angle2]+ TuneSum[:-1][int((angle2+4)%8)])
        angle=angle2
    if IDX<-0.2:
        angle2=angle-2
        IDX = (TuneSum[:-1][angle2]-TuneSum[:-1][int((angle2+4)%8)])/(TuneSum[:-1][angle2]+ TuneSum[:-1][int((angle2+4)%8)])
        if IDX<-0.2: angle=angle+1
        IDX = (TuneSum[:-1][angle]-TuneSum[:-1][int((angle+4)%8)])/(TuneSum[:-1][angle]+ TuneSum[:-1][int((angle+4)%8)])
    
    DG_data = ({'IDX':IDX,'Tuning':TuneSum,'atune':atune,'Rtune':R, 'rasters': ch_raster, 'counts':counts, 'maxcount':maxcount, 'bins':bins})

    ###########################################################
    return TuneSum, atune, R, IDX, counts, maxcount, bins, DG_data