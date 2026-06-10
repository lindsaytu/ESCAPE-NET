# ESCAPE-NET (WINDOWS)
	
*Note: All example datasets referenced in the ESCAPE-NET code are located on the NET drive. This set of instructions will only work on Windows machines.*


## HOW TO RUN:

### DOWNLOAD FILES:

1. Clone repo via `git clone https://github.com/lindsaytu/ESCAPE-NET.git`
2. Modify the file paths in 
- *File_utils_main* (lines 57-83)
- *File_utils* (lines 9, 15)
- *Preprocessing_module_BP* (lines 20, 27, 29)
- *Preprocessing_module_ConPerRing_BP* (lines 21, 28, 30)
  
  to the path of your datasets.

### SETTING UP YOUR ENVIRONMENT:

1. Download the newest version of Anaconda for Windows from https://www.anaconda.com/download
2. Download the conda environment provided ("env.yml")
3. Open Anaconda Prompt
4. Enter the following into Anaconda Prompt to create and activate the environment: 
```
   conda env create -f "env.yml"
   conda activate escape-net
```

### RUN THE CODE:

Enter the following into Anaconda Prompt to run the code:
   ```
   conda activate escape-net
   cd ESCAPE-NET
   python runallCNN_Combined_wfiltsize_BP_Efficient_plus
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

## TROUBLESHOOTING:

The following are some troubleshooting steps for errors that may occur when running the code.

1. MATLAB files that are newer than v7 may not work. To fix this, open a terminal in MATLAB and save with the following command:
```
save('filename.mat', variables, '-v7')
```

2. The ERat label in *runallCNN_Combined_wfiltsize_BP_Efficient_plus.py* (line 26) is based on the labels used in *File_utils_main.py* (lines 57-81), *Preprocessing_module_BP.py* (line 24), and *Preprocessing_module_ConPerRing_BP.py* (line 25). Change this according to the naming convention you set with your datasets.


