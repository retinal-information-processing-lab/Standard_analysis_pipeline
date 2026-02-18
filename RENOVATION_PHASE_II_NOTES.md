# Pipeline renovation - Phase II
## Owners: Chiara, Baptiste

### Get started
1. Ensure you have conda installed (open a terminal and do conda --version)
2. Open a terminal and first create the environment (line 1) and then activate it (line 2), the check your environment has been correctly activated by listing the environments (line 3) and checking something like `...\Standard_analysis_pipeline\env\standard_analysis_pipeline` is in the list. 

```bash
 conda env create -f env/standard_analysis_pipeline.yml --prefix env/standard_analysis_pipeline
 conda activate env/standard_analysis_pipeline
 conda env list
```
3. Now open the notebook you want to use and first ensure you are using the correct python kernel: 
Option 1) add a cell at the beginning of the notebook with the following line of code and run it
    ```python
    !python -m ipykernel install --user --name standard_analysis_pipeline --display-name "standard_analysis_pipeline"
    ```
    you should get something like `Installed kernelspec standard_analysis_pipeline in C:\Users\cboscarino\AppData\Roaming\jupyter\kernels\standard_analysis_pipeline`

Option 2) somewhere in the editor there should be indicated the name of the environment you are using it has to be `standard_analysis_pipeline (Python <version>)`. If it is something like `Python 3.11` or other names, then click on it navigate to `Select another kernel` and chose `standard_analysis_pipeline (Python <version>)`. Once selected the correct kernel you should be able to run the cells of the notebook. 

Note: for running .py files you have might have to select the environment as general interprter for the editor. To change the default interpreter for all Python files in the workspace, procedures might differ according to the IDE. In VScode you can:
Option 1) Click the Python version in the bottom-right corner of VS Code (currently showing something like 3.13.11) and select the standard_analysis_pipeline environment from the list
Option 2) Press Ctrl+Shift+P, search for "Python: Select Interpreter", click "Enter interpreter path" and browse to ...\Standard_analysis_pipeline\env\standard_analysis_pipeline\Scripts\python.exe 
You now should be ready to run any python file in the workspace. 

### Preprocessing
1-Preprocessing-dev.ipynb can be ignored as used only to load test data in phase I. 

We can wait Guilhem is available to do this. 
- What is the difference between 1-New_Preprocessing.ipynb and 1-Preprocessing.ipynb?

### Standard stimuli analysis notebooks
2026-01-26-10:30 (Baptiste) asked claude to do first modularization and cleaning pass on `analyse_checkerboard_steeve.py` -> `analyse_checkerboard_baptiste.py`

2026-01-26-11:12 (Chiara) starting refactoring Checkerboard analysis `analyse_checkerboard_baptiste.py` -> `analyse_checkerboard_chiara.py`: merged analyse_checkerboard_baptiste and first cells of steeve's notebook to check is running before starting modularization. Interrupted due to missing function reference in .py (in steeve and transmitted to baptiste) see issue #28.

2026-01-26-15:30 (Baptiste) Moved prompt_user_for_recording to utils, Prompt parameters not moved, Moved create_analysis_directory to utils, Moved find_Analysis_directory, Moved load_triggers, Moved load_spike_trains, Each notebook has its own specific loader that calls the smaller ones, That should wrap up the loading part, (For notebook 2,3,4,5), Now to raster plots
=> this we may want to move back to the notebook asap

2026-02-10-14:00 (Chiara) Worked on rasters: separating response extraction from plotting and cleaning in checkerboard notebook + .py
    1) analysis.compute_rasters --> analysis.extract_all_cell_responses_to_repeated_sequences + input data types and description + check output data (but same logic)
    2) analysis.plot_rasters --> analysis.plot_all_rasters + input data types and description (but same logic)
    3) analysis.save_plots --> analysis.plot_and_save_single_cell_rasters + input data types and cleaning (removed unused recording number), description (same logic but using plot_raster_and_psth instead of replicated code fragments for plotting)
    4) analysis.plot_one_cell_raster_and_psth --> utils.plot_raster_and_psth and removed repetitive "check each cell" notebook cell

2026-02-18-18:00 (Chiara) Cleaning STA computation, analysis and visualization:
    1) in `utils.extract_from_sequence`: made mandatory `sequence_portion` and `nb_frames_per_sequence` args, args cast and func description
    2) in `utils.compute_3D_sta`: made mandatory `nb_frames_per_sequence` and `temporal_dimension` (to avoid dependence on parms in utils), added args cast and description, check for `min_num_spikes_for_sta` before computation, improved message to user.
    3) adjust all usages of `utils.extract_from_sequence` and `utils.compute_3D_sta` according to mandatory args and adjusted `2-Analyse_Checkerboard_steeve.ipynb` accordingly
    4) adjusted printing, plotting and fig definition in `analyse_checkerboard_steeve.plot_one_cell_3D_spike_triggered_average` 
    5) adjusted sta_analysis: before many analyse_sta versions (Matias, Gabriel, Tom, Guilhem) and `analyse_checkerboard_steeve.fit_ellipse_to_spike_triggered_average` with selection of which one to use. Now `analyse_checkerboard_steeve.analyse_all_stas` is only a wrap for loop on all cells, run compute sta function and save. While sta computation logic is moved in `utils.rf_analysis`, where computation is performed according to selected method/version but common behaviors (like default return in case of failed fitting) are shared: tom and matias method have been merged on common point leaving only different preprocessing according to selected method, guilhem method's logic has been kept, only cleaned default values in case of failed fitting. Also, final check on returned dictionary content added.  
    6) `utils.matias_temporal_spatial_sta` --> `utils.get_temporal_spatial_sta` only modified behavior in case of null sta3d (null sta3d case is handled upstream in `utils.rf_analysis`, thus `utils.get_temporal_spatial_sta` should never be called on an empty sta, so returning none but not handled)
    7) in `utils.plot_sta` added plotting parameters as arguments, and forced blue-red colormap centered on zero
    8) added `analyse_checkerboard_steeve.plot_all_stas` to get an overview of resulting spatial stas upon analysis, added as optional feature in `analyse_all_stas` cell in `2-Analyse_Checkerboard_steeve.ipynb`
    9) Adjusted single-cell sta figures generation: from 2 plots one showing only spatil sta the other spatial sta + fitted elleipse, now 2 plots one showing spatial sta + ellipse the oter temporal sta
    
            



