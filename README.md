# Standard_analysis_pipeline
A repo with all the code you need to go from your raw files to hundreds of fully characterized ganglion cells

Once downloaded the repository all the code runs in python. A standard anaconda distribution with few dependencies will do. The installation of spyking interface is required. The use of jupyter is advised but not necessary. 

If you find any bug or have a feature recommendation, please open a Git issue.

## Installation (2026 version)
1. Ensure you have conda installed (It works for sure with Anaconda 4.11.0, you can find here the ones you need depending on your OS: https://www.anaconda.com/download#downloads) 
(open a terminal and do conda --version)
2. Open a terminal and first create the environment (line 1) and then activate it (line 2), the check your environment has been correctly activated by listing the environments (line 3) and checking something like `...\Standard_analysis_pipeline\env\standard_analysis_pipeline` is in the list.
    ```bash
     conda env create -f env/standard_analysis_pipeline.yml --prefix env/standard_analysis_pipeline
     conda activate env/standard_analysis_pipeline
     conda env list
    ```
    (more on how to do it here: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

3. Spikeinterface for spike sorting is automatically installed with the env (https://spikeinterface.readthedocs.io/en/latest/installation.html). 
For spikeinterface to work hdbscan (https://anaconda.org/conda-forge/hdbscan) and numba (https://numba.pydata.org/numba-doc/latest/user/installing.html) may have to be installed too. Sometimes numba will complain about the version of numpy but reinstalling numpy could create compatibility issues with other packages. Instead, running pip uninstall numba and them pip install -U numba solves it https://stackoverflow.com/questions/74947992/how-to-remove-the-error-systemerror-initialization-of-internal-failed-without.) One may have issues involving Qt plateform plugin "xcb". In this case, please make sure your drivers are up to date. If you still have an issue, check this solution: https://stackoverflow.com/questions/68036484/qt6-qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-thou .

1. Now open the notebook you want to use and first ensure you are using the correct python kernel: 
Option 1) add a cell at the beginning of the notebook with the following line of code and run it
    ```python
    !python -m ipykernel install --user --name standard_analysis_pipeline --display-name "standard_analysis_pipeline"
    ```
    you should get something like `Installed kernelspec standard_analysis_pipeline in C:\Users\cboscarino\AppData\Roaming\jupyter\kernels\standard_analysis_pipeline`

Option 2) somewhere in the editor there should be indicated the name of the environment you are using it has to be `standard_analysis_pipeline (Python <version>)`. If it is something like `Python 3.11` or other names, then click on it navigate to `Select another kernel` and chose `standard_analysis_pipeline (Python <version>)`. Once selected the correct kernel you should be able to run the cells of the notebook. 

Note: for running .py files you have might have to select the environment as general interpreter for the editor. To change the default interpreter for all Python files in the workspace, procedures might differ according to the IDE. In VScode you can:
Option 1) Click the Python version in the bottom-right corner of VS Code (currently showing something like 3.13.11) and select the standard_analysis_pipeline environment from the list
Option 2) Press Ctrl+Shift+P, search for "Python: Select Interpreter", click "Enter interpreter path" and browse to ...\Standard_analysis_pipeline\env\standard_analysis_pipeline\Scripts\python.exe 
You now should be ready to run any python file in the workspace. 


## Additional Dependencies 
- also in the pipeline folder should be present:
  -  the files bynarysource1000Mbits
  -  th SWAN .bin if needed
  -  the probe file for sorting

The analysis of this pipeline are nice and most of all are free. Take advantage of them!