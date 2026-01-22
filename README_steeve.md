
# README-steeve

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
    - Cleaned up notebook `3-Drifting_Gratings-dev.ipynb` and `2-Analyse_Checker...-dev.ipynb` 
        - in first cell: 
            - separate built-in from custom package -> we rarely have to debug built-in packages
            - ideally should contain all paths, parameters input
        - moved notebook functions to associated .py module to enable versioning
    - move long comments on top of code - else, hard to read on small screen


## Unit-tests

- Identified the utils.py functions used in Drifting gratings and checkerboard notebooks
    - load_obj 
        - Drifting gratings [DONE]
        - Checkerboard
    - compute_tuning
        - Drifting gratings
    - save_obj [DONE]-w/ Chiara
        - Drifting gratings
        - Checkerboard    
    - get_recording_spikes [DONE]-w/ Chiara
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

- Setup pytest testing in tests/test_utils.py

## Branches

- This is pushed on a "Develop" branch from which the dev can collaborate
- Devs create their feature branches from "Develop" and pull request/solve conflicts when done.
- At the end, we can automate Github launching of automated testing for Pull requests from "Develop" to "Master".

## Recommendations:

- Experimentalist-only can only interact with notebooks in a notebooks_for_experimentalist/ folder.
    - once, we can move the content of the clean-up, tested functions back to notebook to facilitate their work.
    - they cannot push to the develop , nor main if they modify .py modules.
- Dev. can interact with the entire codebase.