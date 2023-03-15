import os



                                                                     ###################################################################
                                                                     #####################  Experiment Parameters  #####################
                                                                     ###################################################################


#name of your experiment for saving the triggers
exp = r'20230303_MultiSpots'

# select MEA (3=2p room) (4=MEA1 Polychrome)
MEA = 3

#Link to the folder where spiking circus will look for the symbolic links "recording_0i.raw"
symbolic_link_directory = r"/media/samuele/Samuele_2/20230303_MultiSpots/Sorting_3"

#link to .GUI directory where phy extracts all arrays and data on spikes (folder name ends by .GUI)
phy_directory = r'/media/samuele/Samuele_2/20230303_MultiSpots/Sorting_3/recording_00/recording_00.GUI'

#Link to the actual raw files frome the recording listed in the input_file
recording_directory = r"/media/samuele/Samuele_2/20230303_MultiSpots/RAW_Files"

#Link to the directory where output data should be saved
output_directory = r'/media/samuele/Samuele_2/20230303_MultiSpots/Analysis'

#Link to the folder in which triggers will be saved. If doesn't exist, will be created.
triggers_directory = os.path.join(output_directory,"trigs")

#Ordered list of recording_names with your file extension (mostlikly .raw). Don't forget to put it as raw string using r before the name : r'Checkerboard'.
recording_names =    [
"00_CheckerboardAcclim_25D50%_30x20_30Hz.raw",
"01_Checkerboard_25D50%_20x30_30Hz.raw",
"02_Chirp_20reps_25ND50%_50Hz.raw",
"03_DG_25ND50%_2sT_50Hz.raw",
"04_MultiSpots_50reps_25ND50%_40Hz.raw",
"05_MultiSpots_25ND50%_40Hz.raw",
]

binary_source_path = 'binarysource1000Mbits'

registration_directory = r''

#Path to the checkerboard binary file used to generate stimuli
# binary_source_path = ''

# def find_files(path):
#     """
#     Function to get all recording files name from either a txt file name or a folder.

#     Input :
#         - path (string) : a .txt file path containing the recording .raw files name
#         - path (string) : a folder path containing all the recordings .raw files in alphabetic order
        
#     Output :
#         - (list) a list of strings of files names matching the recordings names
        
#     Possible mistakes :
#         - File names are written in .txt without the '.raw' extension
#         - Several files on the same line
#         - Wrong file/folder path
#         - Other files not in '.raw' extension in the folder
#         - Files names aren't ordered
#     """
#     if os.path.isfile(os.path.normpath(path)):                                      #Check if given path is a file and if it exist
#         with open(os.path.normpath(path)) as file:                                      #If yes, than open in variable "file"
#             return file.read().splitlines()                                                 #return the text of each line as a file name ordered from top to bottom
#     return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]       #If no, the path is considered as a folder and return the name of all the files in alphabetic order

# recording_names = find_files(recording_directory) #Do not use this unless you know why ! ! !

                                                                     ###################################################################
                                                                     ####################### Advanced Parameters #######################
                                                                     ###################################################################

"""
    Functions Parameters
    
Default values used in utils functions. If a function has a wrong behaviour, you may want to look in here.
"""

# Datatype used to open rawfiles recordings
dtype = 'uint16'

# Resolution of one step of mea signal amplitude in micro volts
voltage_resolution = 0.1042  # µV / DC level

# Size of a sample in bytes
nb_bytes_by_datapoint=2

# Number of time points to read in a trigger check
probe_size=1000000

# Maximal error admissible in sec for time gap between triggers 
maximal_jitter=0.25e-3

#Checkerboard sequences
nb_frames_by_sequence = 1200

#number of frames to look in for the lag
sta_temporal_dimension = 40

sta_smooth_value = 0.8

sta_treshold = 0.1

temporal_dimension = 30

"""
    SETUP parameters

Only change if you knwo what you are doing. Those parameters are following the setups specs of january 2023
"""

#the optimal threshhold for detecting stimuli onsets varies with the rig
if MEA==1: threshold  = 270e+3         
if MEA==2: threshold  = 50e+3          
if MEA==3: threshold  = 170e+3          
if MEA==4: threshold  = -3.14470e+5

#256 for standard MEA, 17 for MEA1 Polychrome
nb_channels  = 256                

# number of triggers samples acquired per second
fs=20000

#MEA channel id containing holographic triggers trace
holo_channel_id = 127

#MEA channel id containing visual triggers trace
visual_channel_id = 126

# Time before a trigger to remove from the spyking circus analysis due to photo induced current on mea
time_after = 10 #msec

# Time after a trigger to remove  from the spyking circus analysis due to photo induced current on mea
time_before = 10 #msec

# Delay after a trigger to add a fake trigger in the data adding one more dead period 
offset_time = 0.5 #sec

