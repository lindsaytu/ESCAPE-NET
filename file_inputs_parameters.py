"""
Created on June 26, 2026
@author: lindsaytu

This file contains all of the input files from the data collected from the rats. Change based on your data.
There are two training sets, "Training_Sets" and "Training_Sets_ConPerRing". Ensure you have both. This file also contains parameters to change.
"""

class Config:
    def __init__(self):

        self.root_dir = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Lindsay\\'
        self.base_path = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Lindsay\\results\\'
        self.data_path = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Lucie\\31 oct 2022\\right_side\\no_artifact\\new_filter\\Training_Sets\\'
        self.data_path_con_per_ring = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Lucie\\31 oct 2022\\right_side\\no_artifact\\new_filter\\Training_Sets_ConPerRing\\'

        self.datasets = {'ERat1': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\Kwiksil\\',
                           'ERat2': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\',
                           'ERat3': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\no_kwiksil\\',
                           'ERat4': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\',
                           'ERat5': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 23 2021\\Variant F\\',
                           'ERat6': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 23 2021\\Variant F\\no_artifact\\',
                           'ERat7': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\September 9 2021\\Variant F\\no_artifact\\',
                           'ERat8': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\August 27 2021\\no_kwiksil\\no_artifact\\',
                           'ERat9': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\October 7 2021\\Variant F\\no_artifact\\',
                           'ERat10': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\no_artifact\\',
                        'ERat11': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\long_hold\\no_artifact\\',
                        'ERat12': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 9 2021\\Variant F\\longer_hold\\no_artifact\\',
                        'ERat13': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\no_artifact\\',
                        'ERat14': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\session2\\no_artifact\\',
                        'ERat15': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.3_1_1\\no_artifact\\',
                        'ERat16': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.2_1_0.6\\no_artifact\\',
                        'ERat17': 'M:\\Peripheral Nerve Studies\\MCC Projects\\Eugene\\Experiments\\Raw data\\December 17 2021\\Variant F\\ratios_0.2_1_0.7\\no_artifact\\',
    }

        self.num_channels = 64 #number of channels

        self.valid_patience = 15 #early stopping hyperparameter
        self.epochs = 1000 #number of passes of training dataset through model
        self.batch_size = 50 #number of image samples processed together before internal weights are updated
        self.numspikes = 1000 
        self.numfilters_arr = [32]
        self.dense_neurons_arr = [32]#8,4]
        self.filtsizes = [9]
        self.dropout_rate = 0.5 #fraction of neurons randomly dropped during each training step
        self.channel_width_multiplier = 1 #scale factor for size of CNN
        self.print_model = True

config = Config()

import os

if not os.path.exists(config.root_dir):
    print(f"Path doesn't exist: root_dir")

if not os.path.exists(config.base_path):
    print(f"Path doesn't exist: base_path")

if not os.path.exists(config.data_path):
    print(f"Path doesn't exist: data_path")

if not os.path.exists(config.data_path_con_per_ring):
    print(f"Path doesn't exist: data_path_con_per_ring")


for rat, path in config.datasets.items():
    if not os.path.exists(path):
        print(f"Missing Rat: {rat}: {path}")