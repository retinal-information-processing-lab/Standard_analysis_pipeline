
# README-steeve

Major: 

- Added config for conda virtual environment.
- Cleaned up notebook `3-Drifting_Gratings.ipynb` 
    - in first cell: 
        - separate built-in from custom package -> we rarely have to debug built-in packages
    - refactored `import params`
        - moved params' variable to a dictionary -> more readable, editable
        - created small unnested functions to replace huge chunks of code -> more readable
    

Minor:

- move long comment above code - not readable, particularly on small screen/mobile
- ok to keep very small ones on the side