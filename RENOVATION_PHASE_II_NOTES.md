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
    1. analysis.compute_rasters --> analysis.extract_all_cell_responses_to_repeated_sequences + input data types and description + check output data (but same logic)
    2. analysis.plot_rasters --> analysis.plot_all_rasters + input data types and description (but same logic)
    3. analysis.save_plots --> analysis.plot_and_save_single_cell_rasters + input data types and cleaning (removed unused recording number), description (same logic but using plot_raster_and_psth instead of replicated code fragments for plotting)
    4. analysis.plot_one_cell_raster_and_psth --> utils.plot_raster_and_psth and removed repetitive "check each cell" notebook cell

2026-02-18-18:00 (Chiara) Cleaning STA computation, analysis and visualization:
    1. in `utils.extract_from_sequence`: made mandatory `sequence_portion` and `nb_frames_per_sequence` args, args cast and func description
    2. in `utils.compute_3D_sta`: made mandatory `nb_frames_per_sequence` and `temporal_dimension` (to avoid dependence on parms in utils), added args cast and description, check for `min_num_spikes_for_sta` before computation, improved message to user.
    3. adjust all usages of `utils.extract_from_sequence` and `utils.compute_3D_sta` according to mandatory args and adjusted `2-Analyse_Checkerboard_steeve.ipynb` accordingly
    4. adjusted printing, plotting and fig definition in `analyse_checkerboard_steeve.plot_one_cell_3D_spike_triggered_average` 
    5. adjusted sta_analysis: before many analyse_sta versions (Matias, Gabriel, Tom, Guilhem) and `analyse_checkerboard_steeve.fit_ellipse_to_spike_triggered_average` with selection of which one to use. Now `analyse_checkerboard_steeve.analyse_all_stas` is only a wrap for loop on all cells, run compute sta function and save. While sta computation logic is moved in `utils.rf_analysis`, where computation is performed according to selected method/version but common behaviors (like default return in case of failed fitting) are shared: tom and matias method have been merged on common point leaving only different preprocessing according to selected method, guilhem method's logic has been kept, only cleaned default values in case of failed fitting. Also, final check on returned dictionary content added.  
    6. `utils.matias_temporal_spatial_sta` --> `utils.get_temporal_spatial_sta` only modified behavior in case of null sta3d (null sta3d case is handled upstream in `utils.rf_analysis`, thus `utils.get_temporal_spatial_sta` should never be called on an empty sta, so returning none but not handled)
    7. in `utils.plot_sta` added plotting parameters as arguments, and forced blue-red colormap centered on zero
    8. added `analyse_checkerboard_steeve.plot_all_stas` to get an overview of resulting spatial stas upon analysis, added as optional feature in `analyse_all_stas` cell in `2-Analyse_Checkerboard_steeve.ipynb`
    9. Adjusted single-cell sta figures generation: from 2 plots one showing only spatil sta the other spatial sta + fitted elleipse, now 2 plots one showing spatial sta + ellipse the oter temporal sta

2026-02-19-10:00 (Chiara) merged `plot_sta_fitted_with_ellipse_by_tom` to `plot_sta_fitted_with_ellipse`:
    1. added add_raster_plot flag to function to include or not the raster plot in the figure
    2. added ellipse center marker

NEXT: 
make sta figures in physical quantities 
test all and continue with Quantification of the number of STAs (not needed in the standard pipeline) with independent cells (load and process data)

2026-02-23-12:25 (Chiara) final refactoring of checkerboard analysis:
    1. added request for input of `nb_pixels_per_check` with `get_all_inputs_for_checkerboard_analysis` (modified return in `2-Analyse_Checkerboard_steeve.ipynb`, output in `analyse_checkerboard_steeve.get_all_inputs_for_checkerboard_analysis` and `analyse_checkerboard_steeve.prompt_user_for_checkerboard_params` return )
    2. corrected params cast from dict to module type and some others arg cast and returns (icluding stimulus frequency as float instead of int) in both `analyse_checkerboard_steeve` and `utils` (where was given as arg) and corrected some spelling errors
    3. in `analyse_checkerboard_steeve.calculate_checkerboard_experiment_stats`: before as input `stim_onsets: dict, triggers: np.ndarray, ...` but to be used as `calculate_checkerboard_experiment_stats(stim_onsets, stim_onsets, ...)` and useless --> now removed duplicate and used `stim_onsets` (called triggers) to compute everything
    4. fixed actual use of method in `analyse_checkerboard_steeve.analyse_all_stas`, `utils.get_temporal_spatial_sta`
    5. added sta analysis extension to physical unit `analyse_checkerboard_steeve.extend_sta_analysis_to_physical_units` (optional in `2-Analyse_Checkerboard_steeve.ipynb`) and modified `analyse_checkerboard_steeve.plot_sta_fitted_with_ellipse` to report sta details in physical unit if available
    6. adjusted `analyse_checkerboard_steeve.plot_all_stas` with color indicator and added order by property
    7. added RF quantification in `analyse_checkerboard_steeve.plot_sta_fitted_with_ellipse` (text note with spatial rf diameter, area, snr, etc + temporal rf frequency, dealy, etc)
    8. put unused functions at the end in old functions in both `analyse_checkerboard_steeve` and `utils`
    9. added `convert_ellipse_params_to_physical_units`, `get_temporal_sta_time_vector`, `get_cell_delay_time`, `ellipse_area`, `ellipse_radius` and `ellipse_diameter` (added `skimage.measure` to get contour without plt for area computation --> added `scikit-image` to `env\standard_analysis_pipeline.yml`)
    10. `PolyArea` only changed name in `polygon_area`
    11. created `rf_snr`inspired by `SNR_test` 
    12. created `utils.check_rf_fit` 
    13. centered colormap on 0 in `analyse_checkerboard_steeve.plot_one_cell_3D_spike_triggered_average`


TODO (Chiara):
- add the option to save in different format each figure including svg (DONE)
- ADD SCALE BAR EVERYWHERE IN STAs 100 um with note under SNR (DONE)
- add level factor in single cell figures sta (DONE)
- change sta_px in checks in plotting and code (DONE) 
- sigma is not var but std (DONE)

2026-02-27-14:00 (Chiara, Baptiste, Guilhem) standard rf analysis:
    1. normalization of the 3d sta with - median instead of - mean (not changed much in the result)(DONE) 
    2. `preprocess_fitting_standard` as standard function to spatial sta smoothing + noise thresholding inspired from tom preprocessing but with optimized convolution and cleaned thresholding(DONE) 
    3. `get_sta_components` standard function to get spatial and temporal componenets from the 3d sta, using gabriel approach to identify the rf center in time and space (max of the mask, he was using the variance, we switched to the std to avoid narrowing the rf with the square, upon normalization and processing), but spatial and temporal sta returned are slice/trace from unprocessed 3d sta (only normlized inside sta 3d computation)(DONE) 
    4. Added i `rf_analysis` spatial mask and spatial coords of the temporal sta in the data returned, removed plotting in case of failed fitting and added standard method(DONE) 
    5. Added in `plot_sta_fitted_with_ellipse` spatial mask (optional and if available), spatial and temporal components markers in cyan, colorbars for spatial components(DONE) 
    6. explicited correct best_x, best_y in the convention x are cols and y are rows in `get_temporal_spatial_sta` and `matias_temporal_spatial_sta`(DONE) 
    7. compared matias vs tom vs standard fitting methods (`STA_analysis_comparison`) and checked robustness (DONE) 
    8. checked notebook and recomputed analysis with updated computation and standard method (DONE) 
    9. Make more robust the sta computation to not be constrained on sequence portion half (LATER)
    10.use num of sigma instead that levl factor (DONE): This means the RF are now bigger (2 sigmas), previosuly we used 1.35 sigmas

2026-06-26 18:00 (Baptiste) standard vec analysis:
    1. Standardized the vec analysis and move all useful functions to utils
    2. Corrected DG plot to use standard VEC analysis
    3. Corrected Chirp plot to use standard VEC analysis
    4. Upgraded the Typing notebook
        a. Added a function to select DS cell interactively before clustering
        b. This function and the one that passes though STA and Chirps responses now display the images correctly without a need to scroll down
        c. The clustering code is now more robust in case a cell without responses to the chirp was included by mistakes
        d. nonDS and DS cells are now clustered one after the others and clusterID are corrected afterward so that DS cells have the biggest IDs
    5. Cleaning call to params to always vbe explicit, ex: params.x is passed to the functions, never params alone
    6. Moving all utils in a new util folder for clarity and better calls, changed the way functions are called in notebooks to accomodate for this
    7. Addind some unitest to the standard VEC analysis, the rest of the pipeline won't be tested since no one in the lab. is going to use testing correctly

WHAT NEXT?
- ADD OTHER STANDARD STIMULI ANALYSIS (SWaN (Top Priority), Multisize spots, Barcode, MSF)
- REORGANIZE PREPROCESSING + CORRECT CALL TO PARAMS IN PREPROCESSING
- ADD REMI'S CHECK WITH REPEATED STIM (LIKE CHECKERBOARD) TO DO/NOT MERGES DURING SPIKE SORTING
- CORRECT STIM-VIWER TO HELP WITH STIMULUS DESIGN
- ADD THE ANALYSIS OF THE STA THAT TAKES ALSO THE SURROUND SLICE (starting from Olivier's codes?)
