"""
Created on June 18, 2026
@author: lindsaytu

This file contains code that helps run the CNN. The user inputs needed are the locations for the results spreadsheet, filename structure, and number of rats and folds to run.
"""

import os

print(os.path.exists(r"M:\Peripheral Nerve Studies\MCC Projects\Lindsay\results")) #where to save results, verifying that is a valid folder

import pandas as pd
import scipy.io
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
from file_inputs_parameters import config
import ESCAPENET_Efficient_Plus as CNN
import File_utils

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H-%M-%S")

#Don't need to edit this function

def evaluate_model(test_labels, predicted_probs):

    max_probs = np.argmax(predicted_probs, 1)
    one_hot = np.zeros((max_probs.size, 3))
    one_hot[np.arange(max_probs.size), max_probs] = 1

    max_probs = max_probs + 1

    test_labels = np.asarray(test_labels).reshape(-1)
    max_probs = np.asarray(max_probs).reshape(-1)

    # --- Helper functions ---
    def f1(y_true, y_pred):
        y_true = np.eye(3)[y_true - 1]
        tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible = np.sum(np.round(np.clip(y_true, 0, 1)))
        predicted = np.sum(np.round(np.clip(y_pred, 0, 1)))
        recall = tp / (possible + 1e-7)
        precision = tp / (predicted + 1e-7)
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def f1_macro(y_true, y_pred):
        y_true = np.eye(3)[y_true - 1]
        tp = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
        possible = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)
        predicted = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
        recall = tp / (possible + 1e-7)
        precision = tp / (predicted + 1e-7)
        f1_vals = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1_vals, precision, recall

    # --- Metrics ---
    f1score = f1(test_labels, predicted_probs)
    f1score_max = f1(test_labels, one_hot)

    f1_macro_vals, precision_macro, recall_macro = f1_macro(test_labels, predicted_probs)
    f1_macro_mean = np.mean(f1_macro_vals)

    f1_macro_max_vals, precision_macro_max, recall_macro_max = f1_macro(test_labels, one_hot)
    f1_macro_mean_max = np.mean(f1_macro_max_vals)

    accuracy = 1 - np.mean(test_labels != max_probs)

    con_mat = confusion_matrix(test_labels, max_probs)
    con_mat_norm = confusion_matrix(test_labels, max_probs, normalize='true')

    #The scores the model will return, edit only if you do not want all of these

    return {
        "accuracy": accuracy,
        "f1": f1score,
        "f1_max": f1score_max,
        "f1_macro_mean": f1_macro_mean,
        "f1_macro_mean_max": f1_macro_mean_max,
        "conf_mat": con_mat,
        "conf_mat_norm": con_mat_norm
    }

File_utils = File_utils.Utils(config.root_dir,
                                   config.base_path,
                                   config.datasets)

main_dir = config.root_dir
sys.path.append(main_dir)
table = []
columns = []
columns.insert(0, "Host")
#columns.insert(1, "Ratnum")
columns.insert (1, "Model Type")
columns.insert(2, "Fold")
columns.append("Accuracy")
columns.append("F1 score")
columns.append("F1 score max prob")
columns.append("F1 score macro mean")

#Saving results


def save_results(suffix, model_label):
    for host in range(4, 5): #Range of rats to analyze. Make sure this matches below (run the CNN)
        rat_folder = f'ERat{host}\\{config.numfilters_arr[0]}\\'
        for fold in range(1,4): #Number of folds for each rat. Make sure this matches below (run the CNN)
            filename_prefix = (
                f'ERat{host}{suffix}_fold_{fold}'
                f'_filtsize_{config.filtsizes[0]}'
                f'_denselayerdropoutrate_{config.dropout_rate}'
                f'_denseneurons_{config.dense_neurons_arr[0]}'
                f'_conv3dbl_1x1_cwm{config.channel_width_multiplier}'
                f'_rbw' #To read the file. Make sure this matches the filename_prefix below or it will not be able to read the files
            )
            
            full_filename = os.path.join(main_dir, rat_folder, filename_prefix + "_only_conv.mat")
            
            #Check that file exists:
            if not os.path.exists(full_filename):
                print(f"File not found: {full_filename}")
                continue
            
            RAT = scipy.io.loadmat(full_filename)
            test_labels = RAT['test_labels']
            class_probs = RAT['class_probs']
            model_eval = evaluate_model(test_labels, class_probs)
            row = [host,
                model_label,
                #ratnum,
                fold,
                model_eval["accuracy"], #The scores the model will return, edit only if you do not want all of these
                model_eval["f1"], 
                model_eval["f1_max"], 
                model_eval["f1_macro_mean"]
            ]
            table.append(row)

#Run the CNN


for i in range(4,5): #Range of rats to analyze. Make sure this matches above (saving results)
    for foldnum in range(1,4): #Number of folds for each rat. Make sure this matches above (saving results)
        Ratnum = 'ERat' + str(i)
        
        for dense_neurons in config.dense_neurons_arr:
            for numfilters in config.numfilters_arr:
                for filtsize in config.filtsizes:

                    #The filename you want the files to be saved as. Make sure this matches above filename_prefix in saving results

                    rat_folder_main = Ratnum + '\\' + str(numfilters) + '\\'
                    filename_prefix_main = Ratnum + '_fold_' + str(foldnum) + '_filtsize_' + str(filtsize) + '_denselayerdropoutrate_' + str(config.dropout_rate) + '_denseneurons_' + str(dense_neurons) + '_conv3dbl_1x1_cwm' + str(config.channel_width_multiplier) + '_rbw'

                    rat_folder_ConPerRing_main = Ratnum + '\\' + str(numfilters) + '\\'
                    filename_prefix_ConPerRing_main = Ratnum + '_ConPerRing_fold_' + str(foldnum) + '_filtsize_' + str(filtsize) + '_denselayerdropoutrate_' + str(config.dropout_rate) + '_denseneurons_' + str(dense_neurons) + '_conv3dbl_1x1_cwm' + str(config.channel_width_multiplier) + '_rbw'

                    rat_folder_combined_main = Ratnum + '\\' + str(numfilters) + '\\'
                    filename_prefix_combined_main = Ratnum + '_Combined_fold_' + str(foldnum) + '_filtsize_' + str(filtsize) + '_denselayerdropoutrate_' + str(config.dropout_rate) + '_denseneurons_' + str(dense_neurons) + '_conv3dbl_1x1_cwm' + str(config.channel_width_multiplier) + '_rbw'


                    CNN.runCNN_full(Ratnum,foldnum,config.epochs,config.batch_size,config.valid_patience,config.numspikes,numfilters,filtsize,config.dropout_rate,dense_neurons,config.channel_width_multiplier,config.print_model, File_utils, config.data_path, config.num_channels, rat_folder_main, filename_prefix_main, rat_folder_ConPerRing_main, filename_prefix_ConPerRing_main, rat_folder_combined_main, filename_prefix_combined_main)

all_files_exist = True

for i in range(4,5):
    Ratnum = 'ERat' + str(i)

    for foldnum in range(1,4):
        for dense_neurons in config.dense_neurons_arr:
            for numfilters in config.numfilters_arr:
                for filtsize in config.filtsizes:

                    rat_folder = Ratnum + '\\' + str(numfilters) + '\\'

                    filename = (
                        Ratnum + '_Combined_fold_' + str(foldnum)
                        + '_filtsize_' + str(filtsize)
                        + '_denselayerdropoutrate_' + str(config.dropout_rate)
                        + '_denseneurons_' + str(dense_neurons)
                        + '_conv3dbl_1x1_cwm' + str(config.channel_width_multiplier)
                        + '_rbw_only_conv.mat'
                    )

                    full_path = os.path.join(main_dir, rat_folder, filename)

                    if not os.path.exists(full_path):
                        print("Missing File:", full_path)
                        all_files_exist = False


save_results('', 'Temporal')
save_results('_ConPerRing', 'Spatial')
save_results('_Combined', 'Combined')

#Excel sheet for accuracy, f1 values, etc. Change to location you created your results sheet.

results_dir = f'M:\\Peripheral Nerve Studies\\MCC Projects\\Lindsay\\results\\{date_str}'
os.makedirs(results_dir, exist_ok=True)

df = pd.DataFrame(table, columns=columns)
df.to_excel(f'{results_dir}\\results_test_{time_str}.xlsx')
