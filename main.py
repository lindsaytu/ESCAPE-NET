
#save location

#file_utils

import sys
sys.path.append('\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts') #save location (note: all paths below are in windows format, change to linux if needed)

import File_utils_main

File_utils = File_utils_main.Utils('\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts\\Old Image Alignment\\') #save location


#file_utils_main

def data_files():

    new_dataset_mapping = {'ERat1': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\Kwiksil\\',
                           'ERat2': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\',
                           # ERat3 is just ERat2 without kwiksil
                           'ERat3': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\no_kwiksil\\',
                           'ERat4': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\',
                           'ERat5': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 23 2021\\Variant F\\',
                           # ERat6 is ERat5 without artifact
                           'ERat6': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 23 2021\\Variant F\\no_artifact\\',
                                    # M:\Peripheral Nerve Studies\MCC Projects\Eugene\Experiments\Raw data\September 23 2021\Variant F\no_artifact\
                           # ERat7 is ERat 4 without artifact
                           'ERat7': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\no_artifact\\',
                           'ERat8': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\no_kwiksil\\no_artifact\\',
                           'ERat9': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\October 7 2021\\Variant F\\no_artifact\\',
                           
                           'ERat10': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\no_artifact\\',
                                    # M:\Peripheral Nerve Studies\MCC Projects\Eugene\Experiments\Raw data\October 7 2021\Variant F\no_artifact\
                        'ERat11': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\long_hold\\no_artifact\\',
                        'ERat12': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\longer_hold\\no_artifact\\',
                        'ERat13': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\no_artifact\\',
                        'ERat14': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\session2\\no_artifact\\',
                        # ERat14: Changed number of trials to ~50/150/150 tibial/peroneal/sural
                        'ERat15': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.3_1_1\\no_artifact\\',
                        'ERat16': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.2_1_0.6\\no_artifact\\',
                        'ERat17': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.2_1_0.7\\no_artifact\\',
                        # ERat18: First iteration code of oversampling - included test samples in training set so not reliable
                        # - As of Jan 9, corrected using new version of code
    }

#preprocessing module

def preprocessing_filename(self,Ratnum,foldnum,is_RNN=False):

    #change filename to the path of your training set (note: all paths below are in windows format, change to linux if needed)
    load_filename = r'C:\Users\TuL\Documents\MATLAB\Datasets\Training_Fold' + str(foldnum) + '_RAW.mat' #'D:\\Eugene\\Training_Sets\\' + Ratnum + 'Training_Fold' + str(foldnum)
        
    self.numcons = 56
    # Use mapping to get new dataset collected by Eugene
    if "ERat" in Ratnum:
        self.numcons = 64 #number of channels
        if foldnum > 0:
            load_filename = File_utils.new_dataset_mapping[Ratnum] + 'Training_Sets\\Training_Fold' + str(foldnum) + '_RAW'
        else:
            load_filename = File_utils.new_dataset_mapping[Ratnum] + 'Training_Sets\\Dataset_RAW'

#con per ring

def preprocessing_con_per_ring_filename(self, Ratnum, foldnum, is_RNN=False):

    #change filename to path of your ConPerRing training set (note: all paths below are in windows format, change to linux if needed)
        load_filename = 'D:\\Eugene\\Training_Sets_ConPerRing\\' + Ratnum + 'Training_Fold' + str(foldnum)
        
        self.numcons = 56 #number of channels
        # Use mapping to get new dataset collected by Eugene
        if "ERat" in Ratnum:
            self.numcons = 64 #number of channels
            if foldnum > 0:
                load_filename = File_utils.new_dataset_mapping[Ratnum] + 'Training_Sets_ConPerRing\\Training_Fold' + str(foldnum) + '_RAW' #change filename to your path
            else:
                load_filename = File_utils.new_dataset_mapping[Ratnum] + 'Training_Sets_ConPerRing\\Dataset_RAW' #change filename to your path

#runall

import CNN_new_combined_wfiltsize_BP_Efficient_plus as CNN

valid_patience = 15 #early stopping hyperparameter
epochs = 1000 #number of passes of training dataset through model
batch_size = 50 #number of image samples processed together before internal weights are updated
numspikes = 1000 
numfilters_arr = [32]
dense_neurons_arr = [32]#8,4]
filtsizes = [9]
dropout_rate = 0.5 #fraction of neurons randomly dropped during each training step
channel_width_multiplier = 1 #scale factor for size of CNN
# filepath = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Ryan K\\CNNs\\Training_Sets_BP\\'
print_model = True

''' Rats fold 1-3 ''' #change based on number of folds
for i in range(4,11):
    for k in range(1,4):
        ratnum = 'ERat' + str(i)
        
        for dense_neurons in dense_neurons_arr:
            for numfilters in numfilters_arr:
                for filtsize in filtsizes:
                    CNN.runCNN_full(ratnum,k,epochs,batch_size,valid_patience,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,channel_width_multiplier,print_model)