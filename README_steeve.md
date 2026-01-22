
# README-steeve


Test dataset used (from Guilhelm): 20251219_PulsingGratings_PupilSize/

Issues:

- Large chunks of code 
    - very hard to debug -> refactor code to make readable, understandable.
    - notebook too long -> endless scrolling
- Control of the variables in the environment at anytime must be improved.
- The code must be modularized for tested/validation/debugging/sharing.

Major: 

Clean up:
- Added config for conda virtual environment.
- Cleaned up notebook `3-Drifting_Gratings.ipynb` and `2-Analyse_Cheker....ipynb` 
    - in first cell: 
        - separate built-in from custom package -> we rarely have to debug built-in packages
        - this should contain all paths, parameters input
    - refactored `import params`
        - moved params' variable to a dictionary -> more readable, editable
        - created small unnested functions to replace huge chunks of code -> more readable
    - all functions should be moved to a .py module for versioning

Unit-testing:
- Identified the utils functions used in Drifting gratings and checkerboard notebook


Minor:

- move long comment above code - not readable, particularly on small screen/mobile
- ok to keep very small ones on the side


Recommendations:

- Experimentalist-only should only interact with notebooks in a notebooks_for_experimentalist/ folder.
    - they can copy paste functions from utils to their notebooks
    - they can never modify utils.py; if they do they should not be allowed to push to the master branch.
- Dev can interact with the entire codebase