
# README-steeve


Test dataset used (from Guilhelm): 20251219_PulsingGratings_PupilSize/

## Refactoring

- Large chunks of code 
    - very hard to debug -> refactor code to make readable, understandable.
    - notebook too long -> endless scrolling
- Control of the variables in the environment at anytime must be improved.
- The code must be modularized for tested/validation/debugging/sharing.

Major: 

Clean up:

- Added config for conda virtual environment.
- Cleaned up notebook `3-Drifting_Gratings-dev.ipynb` and `2-Analyse_Checker...-dev.ipynb` 
    - refactored `import params`
        - moved params' variable to a dictionary -> more readable, editable
        - created small unnested functions to replace huge chunks of code -> more readable
    - in first cell: 
        - separate built-in from custom package -> we rarely have to debug built-in packages
        - ideally should contain all paths, parameters input
    - I moved notebook functions to a .py module for versioning

Minor:

- move long comment above code - hard to read on small screen

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

## Unit-tests

- Identified the utils.py functions used in Drifting gratings and checkerboard notebook
    - load_obj 
        - Drifting gratings [DONE]
        - Checkerboard
    - compute_tuning
        - Drifting gratings
    - save_obj [TODO]-w/ Chiara
        - Drifting gratings
        - Checkerboard    
    - get_recording_spikes [TODO]-w/ Chiara
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
    

## Recommendations:

- Experimentalist-only can interact with notebooks in a notebooks_for_experimentalist/ folder.
    - once, we can move the content of the clean-up, tested functions back to notebook to facilitate their work.
    - they cannot push to the develop , nor main if they modify .py modules.
- Dev. can interact with the entire codebase.