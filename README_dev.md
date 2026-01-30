# README-developer

# Step 1

List of completed tasks.

## Test dataset used

Test dataset used (from Guilhelm): 20251219_PulsingGratings_PupilSize/

## Installation 

### Setup the conda virtual environment

TODO!: env/standard_analysis_pipeline.yml must be updated with all the dependencies

Requirements: > conda 23.3.1

Move to the root of the repository and install:

```bash
conda env create -f env/standard_analysis_pipeline.yml --prefix env/standard_analysis_pipeline
conda activate env/standard_analysis_pipeline
```

Execution time < 1 min

Tested on Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K ＠3.2 GHz/5.8 GHz)


## Basic refactoring

- Large chunks of code 
    - very hard to debug -> refactor code to make readable, understandable.
    - notebook too long -> endless scrolling
    - Use functions:
        - have control of the variables in the environment at anytime.
        - modular
        - enables testing/validation/debugging/sharing.

- Clean up:
    - Added .yml config. for conda virtual environment.
    - Refactored `import params`:
        - moved params' variable to a dictionary -> more readable, editable
        - created small un-nested functions to replace huge chunks of code -> more readable    
    - Cleaned up notebook 
        - `3-Drifting_Gratings_steeve.ipynb`
        - `2-Analyse_Checkerboard_steeve.ipynb` 
        - in first cell: 
            - separate built-in from custom package -> we rarely have to debug built-in packages
            - ideally should contain all paths, parameters input
        - moved notebook functions to associated .py module to enable versioning
    - Create associated modules: 
        - `drifting_gratings_steeve.py`
        - `analyse_checkerboard_steeve.py`
    - move long comments on top of code - else, hard to read on small screen


## Unit-tests

- Identified the utils.py functions used in Drifting gratings and checkerboard notebooks
    - `load_obj`
        - Drifting gratings [DONE]
        - Checkerboard
    - compute_tuning
        - Drifting gratings
    - `save_obj` [DONE]-w/ Chiara
        - Drifting gratings
        - Checkerboard    
    - `get_recording_spikes` [DONE]-w/ Chiara
       - Checkerboard    
    - extract_from_sequence [TODO] - requires logic
        - Checkerboard    
    - compute_3D_sta [TODO] - requires logic
        - Checkerboard    
    - analyse_sta_tom [TODO] - requires logic
        - Checkerboard    
    - analyse_sta_matias [TODO] - requires logic
        - Checkerboard    
    - analyse_sta_gab [TODO] - requires logic
        - Checkerboard    
    - analyse_sta [TODO] - requires logic
        - Checkerboard    

- Setup pytest testing in `tests/test_utils.py`

## Branches

- This is pushed on a `"Develop"` branch from which the dev can collaborate
- Devs create their feature branches from "Develop" and pull request/solve conflicts when done.
- At the end, we can automate Github launching of automated testing for Pull requests from "Develop" to "Master".

## Recommendations:

- Experimentalist-only can only interact with notebooks in a notebooks_for_experimentalist/ folder.
    - once, we can move the content of the clean-up, tested functions back to notebook to facilitate their work.
    - they cannot push to the develop , nor main if they modify .py modules.
- Dev. can interact with the entire codebase.

# Step 2

## Baptiste's note

### What I did

I included and tested the following function
- load_spike_times(params, rec, verbose = False): Load spike times for all neurons for a given recording from the fullexp_neurons_data.pkl file.
        Helpful to help reducing the liens it takes to read data from fullexp.pkl

- load_stim_onset_from_triggers_path(triggers_path, params, verbose: bool ) : Load trigger data from saved file and give the stim onset aleady converted in second.
        Helpful for getting stim_onsets (the real thing we care about) from raw trigger data

- prompt_user_for_recording(params: dict, stim_name: str) : Help with showing the prompt that helps findinf the .vec

- create_analysis_directory(params: dict, recording_number: int, analysis_name: str): Create the calssicale directories used by the pipeline analysis

- find_analysis_directory(dir_type="Checkerboard", output_directory = params.output_directory): Look for specific analysis directory

Those changes have been implemented up to the notebook (which I tested)

### Note on Utils

- There are still some magic variables (fixed params in CAPITAL letters) that should be removed
- Except for the analyse_STA (with all the named versions), the gaussian fitting and the preprocessing, there is nearly no duplciation in utils.
- All functions should be changed to use snake_case.

- I'm still not super conviced about the specific analysis .py, I really think we'll want to move them back to the notebooks at some point (trade off between nice for experimentalist and easy to debug).

### Noteds PSTH + RASTER ANALYSIS

This is just a sketch of functions tha could be useful.
Some of it is already handled in utils.py in build_rasters or etract_from_sequence (+ others).
But it needs to be be slightly reworked to be more general and reusable.
Overall the notebook 'Standard_Vec_Analysis' should also contain important ideas here!
The accent should also be made on separating low-level stuff (data handling, I/O) from higher level stuff (plots).

def compute_rasters(spikes: dict, triggers: np.ndarray, 
                   nb_repeats: int, stimulus_frequency: int) -> dict:
    """
    Compute raster data for all cells.
    
    NOTE : Redundant with build_rasters
    
    Args:
        checkerboard_spikes: Dict mapping cell IDs to spike times
        triggers: Array of trigger times
        nb_repeats: Number of stimulus repetitions
        stimulus_frequency: Stimulus frequency in Hz
        
    Returns:
        Dictionary mapping cell IDs to raster data
    """
    print('Computing rasters...')
    raster_data = {}
    
    for cell_id, spike_times in tqdm(spikes.items()):
        raster_data[cell_id] = extract_from_sequence(
            spike_times, triggers, nb_repeats, stim_frequency=stimulus_frequency
        )
    
    return raster_data

def plot_single_raster(ax, spike_trains, color='darkblue', linelength=0.8):
    """
    Plot raster for a single cell.
    From data directly.

    Args:
        ax: Matplotlib axis 
        spike_trains: List of spike trains
    """
    ax.eventplot(spike_trains, colors=color, linelengths=linelength)
    ax.set(title="Raster plot", ylabel="N Repetitions", xlabel="Time in sec",)

def plot_single_psth_from_raster_data(ax, raster_data, cell_nb, params: dict):
    """
    Plot PSTH for a single cell.
    From raster data directly to extract all information automatically.

    Args:
        ax: Matplotlib axis
        raster_data: Dictionary with raster data
        cell_nb: Cell number to plot
        params: Dictionary with 'nb_frames_by_sequence'
    """
    width = raster_data[cell_nb]["repeated_sequences_times"][0][0] / int(
        params.nb_frames_by_sequence / 2
    )
    seq_length = (
        raster_data[cell_nb]["repeated_sequences_times"][0][1]
        - raster_data[cell_nb]["repeated_sequences_times"][0][0]
    )

    x_vals = (
        np.linspace(0, seq_length, int(params.nb_frames_by_sequence / 2)) + width / 2
    )
    ax.bar(x_vals, raster_data[cell_nb]["psth"], width=1.3 * width)
    ax.set(xlabel="Time in sec", ylabel="Firing rate (spikes/s)")