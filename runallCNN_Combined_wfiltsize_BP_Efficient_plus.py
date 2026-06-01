# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:00:22 2018

@author: kohr
"""

import CNN_new_combined_wfiltsize_BP_Efficient_plus as CNN

valid_patience = 15
epochs = 1000
batch_size = 50
numspikes = 1000
numfilters_arr = [32]
dense_neurons_arr = [32]#8,4]
#filtsize = 8
filtsizes = [9]
dropout_rate = 0.5
channel_width_multiplier = 1
# filepath = 'M:\\Peripheral Nerve Studies\\MCC Projects\\Ryan K\\CNNs\\Training_Sets_BP\\'
print_model = True

''' Rats fold 1-3 '''
for i in range(4,11):
    for k in range(1,4):
        ratnum = 'ERat' + str(i) #ratnum = 'Rat' + str(i)
        
        for dense_neurons in dense_neurons_arr:
            for numfilters in numfilters_arr:
                for filtsize in filtsizes:
                    CNN.runCNN_full(ratnum,k,epochs,batch_size,valid_patience,numspikes,numfilters,filtsize,dropout_rate,dense_neurons,channel_width_multiplier,print_model)

               