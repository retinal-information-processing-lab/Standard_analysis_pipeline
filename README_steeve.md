
# README-steeve


Test dataset used (from Guilhelm): 20251219_PulsingGratings_PupilSize/

## Issues

- Large chunks of code 
    - very hard to debug -> refactor code to make readable, understandable.
    - notebook too long -> endless scrolling
- Control of the variables in the environment at anytime must be improved.
- The code must be modularized for tested/validation/debugging/sharing.

Major: 

Clean up:
- Added config for conda virtual environment.
- Cleaned up notebook `3-Drifting_Gratings-dev.ipynb` and `2-Analyse_Checker...-dev.ipynb` 
    - in first cell: 
        - separate built-in from custom package -> we rarely have to debug built-in packages
        - this should contain all paths, parameters input
    - refactored `import params`
        - moved params' variable to a dictionary -> more readable, editable
        - created small unnested functions to replace huge chunks of code -> more readable
    - all functions should be moved to a .py module for versioning

Minor:

- move long comment above code - not readable, particularly on small screen/mobile
- ok to keep very small ones on the side

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
    - once, we can move the content of the clean-up, tested functions back to notebook to facilitate
    - they can never modify utils.py; if they do they should not be allowed to push to the master branch.
- Dev can interact with the entire codebase