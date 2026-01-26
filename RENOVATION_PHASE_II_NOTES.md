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

NOTES: 
1. the existing environment .yml cannot be created as it is because it says there is a conflict between python version (python 10) and the one required by spikeinterface=0.103.2 (python >11/13). 

```Could not solve for environment specs
The following packages are incompatible
├─ python =3.10.16 * is requested and can be installed;
└─ spikeinterface =0.103.2 * is not installable because there are no viable options
   ├─ spikeinterface 0.103.2 would require
   │  └─ python >=3.11 * but there are no viable options
   │     ├─ python [3.11.0|3.11.1|...|3.12.9] conflicts with any installable versions previously reported; 
   │     ├─ python [3.13.0|3.13.1|...|3.14.2] conflicts with any installable versions previously reported; 
   │     └─ python [3.14.0rc1|3.14.0rc2|3.14.0rc3] would require
   │        └─ _python_rc =* *, which does not exist (perhaps a missing channel);
   └─ spikeinterface 0.103.2 would require
      └─ python >=3.13 * but there are no viable options
         ├─ python [3.13.0|3.13.1|...|3.14.2] conflicts with any installable versions previously reported; 
         └─ python [3.14.0rc1|3.14.0rc2|3.14.0rc3], which cannot be installed (as previously explained). 
```

However spiking interface is used only in preprocessing->sorting, we can do a pipeline that does not depends on spiking interface (and install it only for required portions). So so far we just commented the spiking interface line in the environment and it worked. 

### Preprocessing
1-Preprocessing-dev.ipynb can be ignored as used only to load test data in phase I. 

We can wait Guilhem is available to do this. 
- What is the difference between 1-New_Preprocessing.ipynb and 1-Preprocessing.ipynb?

### Standard stimuli analysis notebooks
#### 2-Analyse_Checkerboard_steeve.ipynb
