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
2026-01-26-11:12 (Chiara) starting refactoring Checkerboard analysis `analyse_checkerboard_baptiste.py` -> `analyse_checkerboard_chiara.py`
