# ESCAPE-NET (WINDOWS)
	
Extraneural Spatiotemporal CAPs Extraction Network (ESCAPE-NET) is a method of using a convolutional neural network (CNN) to classify naturally-evoked compound action potentials (nCAPs). This is a guide for running ESCAPE-NET on a Windows machine with your datasets. *Note: All example datasets referenced in the ESCAPE-NET code are located on the NET drive. This set of instructions will only work on Windows machines.*


## HOW TO RUN:

### DOWNLOAD FILES:

1. Open a PowerShell terminal and install git if it is not already installed via the following command:
```
winget install --id Git.Git -e --source winget
```
2. Clone repo via `git clone https://github.com/lindsaytu/ESCAPE-NET.git`
3. Modify the file paths in 
- [File_utils_main](File_utils_main) (lines 57-83)
- [File_utils](File_utils) (lines 9, 15)
- [Preprocessing_module_BP](Preprocessing_module_BP) (lines 20, 27, 29)
- [Preprocessing_module_ConPerRing_BP](Preprocessing_module_ConPerRing_BP) (lines 21, 28, 30)
  
  to the path of your datasets.

### SETTING UP YOUR ENVIRONMENT:

1. Download the newest version of Anaconda for Windows from https://www.anaconda.com/download
2. Have the conda environment file ("env.yml") ready. This file is included when you clone the repo.
3. Open Anaconda Prompt
4. Enter the following into Anaconda Prompt to create and activate the environment: 
```
cd "C:\Users\<username>\Desktop\ESCAPE-NET" #change path to where you cloned the repo
conda env create -f "env.yml"
conda activate escape-net
```
Note: it may take a few minutes to create the conda environment. The following is an example of the successful output:

<img width="553" height="177" alt="Screenshot 2026-06-12 at 11 18 23 AM" src="https://github.com/user-attachments/assets/d478c2bc-a94d-4822-bbf3-ba0b6d503dda" />

### RUN THE CODE:

After setting up the environment, enter the following into Anaconda Prompt to run the code:
   ```
   python runallCNN_Combined_wfiltsize_BP_Efficient_plus.py
   ```

## PARAMETERS

These are the final parameters that were used in Eugene's paper for ESCAPE-NET Mini with four convolutional layers. These are defined in [runallCNN_Combined_wfiltsize_BP_Efficient_plus.py](runallCNN_Combined_wfiltsize_BP_Efficient_plus.py):
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

2. The ERat label in [runallCNN_Combined_wfiltsize_BP_Efficient_plus.py](runallCNN_Combined_wfiltsize_BP_Efficient_plus.py) (line 26) is based on the labels used in [File_utils_main.py](File_utils_main.py) (lines 57-81), [Preprocessing_module_BP.py](Preprocessing_module_BP.py) (line 24), and [Preprocessing_module_ConPerRing_BP.py](Preprocessing_module_ConPerRing_BP.py) (line 25). Change this according to the naming convention you set with your datasets.


