# ESCAPE-NET (WINDOWS)
	
*Note: All example datasets referenced in the ESCAPE-NET code are located on the NET drive. This set of instructions will only work on Windows machines.*

## SETTING UP YOUR ENVIRONMENT:

1. Download the newest version of Anaconda for Windows from https://www.anaconda.com/download
2. Download the conda environment provided ("env.yml")
3. Open the environment file and update the first and last lines
   - First line: name: *name of your environment*
     * Example:
       ```name: environment```
   - Last line: prefix: *full path to your environment*
     * Example:
       ```prefix: C:\Users\TuL\AppData\Local\anaconda3\envs\environment ```
4. Open Anaconda Prompt
5. Enter the following into Anaconda Prompt to create and activate the environment: 
```
   conda env create -f "path to your environment"
   conda activate *name of your environment*
```
## PARAMETERS

These are the final parameters that were used in Eugene's paper for ESCAPE-NET Mini with four convolutional layers. These are defined in runallCNN_Combined_wfiltsize_BP_Efficient_plus.py:
```
numfilters = 4 #convolutional layer width
dense_neurons_arr = [8] #dense layer width
filtsizes = [8] #filter size of first layer. note: other layers are 4x4, 2x2, 1x1 but are hardcoded into CNN_new_combined_wfiltsize_BP_Efficient_plus.py
dropout_rate = 0
epochs = 25
```

## HOW TO RUN:

1. Download the python files provided
2. Modify the file paths in *File_utils_main*, *File_utils*, *Preprocessing_module_BP*, and *Preprocessing_module_ConPerRing_BP* to link to your datasets. 
3. Enter the following into Anaconda Prompt to run the code:
   ```
   conda activate *name of your environment*
   cd *name of directory the code is stored in*
   python runallCNN_Combined_wfiltsize_BP_Efficient_plus
   ```


## COMMON ISSUES & CHANGES:

1. MATLAB files that are newer than v7 may not work. To fix this, open a terminal in MATLAB and save with the following command:
```
save('filename.mat', variables, '-v7')
```

2. The ERat label in *runallCNN_Combined_wfiltsize_BP_Efficient_plus.py* is based on the labels used in *File_utils_main.py*, *Preprocessing_module_BP.py*, and *Preprocessing_module_ConPerRing_BP.py*. Change this according to the naming convention you set with your datasets.
