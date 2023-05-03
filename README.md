# Standard_analysis_pipeline
A repo with all the code you need to go from your raw files to hundreds of fully characterized ganglion cells


Once downloaded the repository all the code runs in python. A standard anaconda distribution with few dependencies will do. The installation of spyking interface is required. The use of jupyter is advised but not necessary. 

## Installation
Once you have your Anaconda distribution for python (you can find here the ones you need depending on your OS: https://www.anaconda.com/download#downloads) you need to create an environment (here how to do it: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) and to install spikeinterface in it (here how to do it: https://spikeinterface.readthedocs.io/en/latest/installation.html). 
It works for sure with Anaconda 4.11.0 and python 3.9.16 but also later versions of anaconda should be compatible. For spikeinterface to work hdbscan (https://anaconda.org/conda-forge/hdbscan) and numba (https://numba.pydata.org/numba-doc/latest/user/installing.html) have to be installed. Sometimes numba will complain about the version of numpy but reinstalling numpy could create compatibility issues with other packages. Instead, running pip uninstall numba and them pip install -U numba solves it https://stackoverflow.com/questions/74947992/how-to-remove-the-error-systemerror-initialization-of-internal-failed-without. 

## Dependencies
- colorama: https://pypi.org/project/colorama/
- scikit-image: https://scikit-image.org/docs/stable/install.html
- PyQt5
- also in the pipeline folder should be present:
  -  the files bynarysource1000Mbits
  -  the Chirp vecs for plotting
  -  the probe file for sorting

The analysis of this pipeline are nice and most of all are free. Take advantage of them!
