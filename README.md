# ESCAPE-NET (WINDOWS)
	
Note: All example datasets referenced in the ESCAPE-NET code are located on the NET drive. This set of instructions will only work on Windows machines. Please consult the Linux guide for Linux instructions. 

## SETTING UP YOUR ENVIRONMENT:

1. Download the newest version of Anaconda for Windows from https://www.anaconda.com/download
2. Download the conda environment provided in Peripheral Nerve Studies/MCC/Arthur/environment/env on the NET Drive
3. Open the environment file and update the first and last lines
   - First line: name: *name of your environment*
     * Example: name: environment
   - Last line: prefix: *full path to your environment*
     * Example: prefix: C:\Users\TuL\AppData\Local\anaconda3\envs\environment 
5. Open Anaconda Prompt
6. Enter the following into Anaconda Prompt to create and activate the environment: 
```
   conda env create -f "path to your environment"
   conda activate *name of your environment*
```
## HOW TO RUN:

1. Modify the file paths in File_utils_main, File_utils, Preprocessing_module_BP, and Preprocessing_module_ConPerRing_BP to link to your datasets. 
2. Enter the following into Anaconda Prompt to run the code:
   ```
   conda activate *name of your environment*
   cd *name of directory the code is stored in*
   python runallCNN_Combined_wfiltsize_BP_Efficient_plus
   ```

Your code should be up and running now! 
