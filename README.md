# ESCAPE-NET (WINDOWS)
	
Extraneural Spatiotemporal CAPs Extraction Network (ESCAPE-NET) is a method of using a convolutional neural network (CNN) to classify naturally-evoked compound action potentials (nCAPs). This is a guide for running ESCAPE-NET on a Windows machine with your datasets. *Note: All example datasets referenced in the ESCAPE-NET code are located on the NET drive. This set of instructions will only work on Windows machines.* 


## HOW TO RUN:

### DOWNLOAD & MODIFY FILES:

1. Open a PowerShell terminal and install git if it is not already installed via the following command:
```
winget install --id Git.Git -e --source winget
```
2. Clone repo via `git clone https://github.com/lindsaytu/ESCAPE-NET.git`
3. There are two files you have to modify:
    - main.py: In this file, add your results spreadsheet, your filename naming structure, and change the number of rats and folds you want to run.
    - file_inputs_parameters.py: In this file, add the paths to your training sets, datasets, and change parameters for the CNN. See section "PARAMETERS" for default parameters to use.

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
   python ESCAPENET_Efficient_Plus.py
   ```

## PARAMETERS

These are the final parameters that were used in Eugene's paper for ESCAPE-NET Mini with four convolutional layers. These are defined in [file_inputs_parameters.py](file_inputs_parameters.py):
```
num_channels = 56
numfilters = 4 #convolutional layer width
dense_neurons_arr = [8] #dense layer width
filtsizes = [8] #filter size of first layer. note: other layers are 4x4, 2x2, 1x1 but are hardcoded into ESCAPENET_Efficient_Plus.py
dropout_rate = 0
epochs = 25
```

## TROUBLESHOOTING:

The following are some troubleshooting steps for errors that may occur when running the code.

1. The ERat label in [main.py](main.py) (lines 101, 104, 137) is based on the labels used in [file_inputs_parameters.py](file_inputs_parameters.py) (lines 17-33)If you run into an error, it might be because you changed one label without changing the others. Double check all the files to make sure the naming convention is consistent.


