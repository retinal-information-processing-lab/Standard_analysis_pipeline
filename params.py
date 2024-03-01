import os
                                                                 ###################################################################
#####################  Experiment Parameters  #####################
###################################################################
"""
    Experiment Parameters
    
Various names of folders and files that you need to set up for the pipeline to work.
"""
#### ----------------------- ALWAYS check these names ----------------------- ####


root = r"/home/guiglaz/Documents/Pipeline_Dev"   # Root folder of your experiment
                                                      # all other files must be inside of this folder or manually specified.
            
exp = r'20231125_VIP.Project_VDH--FF+Disk_100um(dark)'  #name of your experiment for saving the triggers

MEA = 3                # select MEA (3=2p room) (4=MEA1 Polychrome)

raw_files_folder = r"RAW_Files"  #Enter the name of the folder containing all your raw files. 
                                 #It will be conctenated with root to find your raws. 
                                 #If the folder is not in root, change the variable "recording_directory" manually.


#Ordered list of recording_names without your file extension (mostlikly .raw). Don't forget to put it as raw string using r before the name : r'Checkerboard'.
recording_names =    []

registration_directory = r'20231125_VIP.Project_FF+Disk_100um_DHguiOptimization'  #Set the registration folder name here.
                              #If you don't know what that is keep it empty otherwise.

## Your MEA parameters
    
mea_spacing = 30            # the sapcing between two electrodes of the MEA in µm for registration

n_electrodes = 16           # number of electrodes on one side of the MEA. N tot electrodes = n_electrodes**2



# Unless you have a specific file organisation, you don't need to change anything bellow this line

#### ----------------------- Automatic folders creation ----------------------- ####

#Link to the actual raw files frome the recording listed in the input_file
recording_directory = os.path.join(root,raw_files_folder)

#Link to the folder where spiking circus will look for the symbolic links "recording_0i.raw"
symbolic_link_directory = os.path.join(root,r"Sorting")
if not os.path.exists(symbolic_link_directory): os.makedirs(symbolic_link_directory)

sorting_directory = symbolic_link_directory

#link to .GUI directory where phy extracts all arrays and data on spikes (folder name ends by .GUI)
phy_directory = os.path.normpath(os.path.join(symbolic_link_directory, r'recording_00/recording_00.GUI'))

#Link to the directory where output data should be saved
output_directory = os.path.join(root,r'Analysis')
if not os.path.exists(output_directory): os.makedirs(output_directory)

#Link to the folder in which triggers will be saved. If doesn't exist, will be created.
triggers_directory = os.path.join(output_directory,"triggers")
if not os.path.exists(triggers_directory): os.makedirs(triggers_directory)

#Path to the checkerboard binary file used to generate stimuli
binary_source_path = 'binarysource1000Mbits'

raw_filtered_directory = os.path.join(root,'RAW_filtered')

registration_frames = os.path.join(root, registration_directory+r"/frames/")

registration_imgs = os.path.join(root, registration_directory+r"/imgs/")


# Automatic raw files detection
def find_files(path):
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
    if os.path.isfile(os.path.normpath(path)):                                      #Check if given path is a file and if it exist
        with open(os.path.normpath(path)) as file:                                      #If yes, than open in variable "file"
            return file.read().splitlines()                                                 #return the text of each line as a file name ordered from top to bottom
    return sorted([os.path.splitext(f)[0] for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == ".raw")])       #If no, the path is considered as a folder and return the name of all the files in alphabetic order

recording_names = find_files(recording_directory) #Do not use this unless you know how  ! ! !

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

# Time in s at the begining of the recording used to check recording type
time = 10

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
if MEA==3:
    threshold  = 170e+3   
    size_dmd = [760, 1020]      # dimensions of the DMD, in pixels
    pxl_size_dmd = 2.5          # The size of one pixel of the DMD in µm? on the camera or in reality?

if MEA==4: threshold  = -3.14470e+5

#256 for standard MEA, 17 for MEA1 Polychrome
nb_channels  = 256                

#MEA channel id containing holographic triggers trace
holo_channel_id = 127

#MEA channel id containing visual triggers trace
visual_channel_id = 126

# number of triggers samples acquired per second
fs=20000

# Time before a trigger to remove from the spyking circus analysis due to photo induced current on mea
time_after = 10 #msec

# Time after a trigger to remove  from the spyking circus analysis due to photo induced current on mea
time_before = 10 #msec

# Delay after a trigger to add a fake trigger in the data adding one more dead period 
offset_time = 0.5 #sec

"""
    Pipeline params
"""

ressources = r"./ressources" #relative path from pipeline notebook to a folder containing ressources such as mea pictures and datasets

