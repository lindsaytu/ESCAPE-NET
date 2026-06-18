# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:53:20 2024

@author: SunY
"""
import sys
import pandas as pd
import os.path
import pathlib
from os import path
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
from openpyxl.workbook import Workbook
from collections import OrderedDict
import scipy.io
from datetime import datetime #add date

main_dir = 'M:\\NET2\\Zariffa\\Arthur\\Eugene\\NN Scripts\\Old Image Alignment\\'
sys.path.append(main_dir)
table = []
columns = []
columns.insert(0, "Host")
#columns.insert(1, "Ratnum")
columns.insert(1, "Fold")
columns.append("Accuracy")
columns.append("F1 score")
columns.append("F1 score max prob")
columns.append("F1 score macro mean")

now = datetime.now
date_str = now.strftime("%Y-%m-%d")

for host in range(83, 92):
    rat_folder = 'ERat'+ str(host) + '_' + date_str
    #for ratnum in range (83, 92):
    for fold in range(1,4):
        #full_filename = main_dir + rat_folder + 'ERat'+str(host)+'_DF_PF_Prick_wnoise_CM_CM_CDD_Combined_fold'+str(fold)+'_filtsize_8_denselayerdropoutrate_0_denseneurons_32_cwm1_numlayer4_tlNEWERat'+str(ratnum)+'_only_conv.mat'
        full_filename = main_dir + rat_folder + 'ERat' + str(host) + '_DF_PF_Prick_wnoise_CM_CM_CDD_Combined_fold'+str(fold)+'_filtsize_8_denselayerdropoutrate_0_denseneurons_32_cwm1_numlayer4_only_conv.mat'
        RAT = scipy.io.loadmat(full_filename)
        test_labels = RAT['test_labels']
        class_probs = RAT['class_probs']
        model_eval = utils.evaluate_model(test_labels, class_probs)
        row = [host,
               #ratnum,
               fold,
               model_eval.accuracy, 
               model_eval.f1score, 
               model_eval.f1score_maxprobs, 
               model_eval.f1score_macro_mean]
        table.append(row)
            
df = pd.DataFrame(table, columns=columns)
df.to_excel('M:\\NET2\\Zariffa\\Arthur\\Eugene\\NN Scripts\\Old Image Alignment\\results\\transfer_learning_results_hosts.xlsx')
    