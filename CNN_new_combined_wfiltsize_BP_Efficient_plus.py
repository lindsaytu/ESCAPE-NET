# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:09:27 2018

@author: kohr

"""

import Preprocessing_module_BP as PP
import Preprocessing_module_ConPerRing_BP as PP2
import numpy as np
import scipy
from main import (
    File_utils, rat_folder_main, filename_prefix_main, rat_folder_ConPerRing_main, filename_prefix_ConPerRing_main, rat_folder_combined_main, filename_prefix_combined_main
)

import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD #from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import tensorflow as tf 

def runCNN_full(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model):
    
    print(Ratnum + ': Fold ' + str(foldnum) + ' starting')
    
    [Int_model_sp,RAT_DATA_sp] = CNN_CM_CM_DD(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model)
        
    print(Ratnum + ': Fold ' + str(foldnum) + ' spatial_done')
    
    Int_model_tp = None
    RAT_DATA_tp = None
    # if print_model:
    #     [Int_model_tp,RAT_DATA_tp] = CNN_CM_CM_DD_ConPerRing(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,None, \
    #                                                          None,None,numfilters,filtsize, dropout_rate,dense_neurons,print_model)
    
    [Int_model_tp,RAT_DATA_tp] = CNN_CM_CM_DD_ConPerRing(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,RAT_DATA_sp.samples1, \
                                                         RAT_DATA_sp.samples2,RAT_DATA_sp.samples3,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model)

    print(Ratnum + ': Fold ' + str(foldnum) + ' temporal_done')

    combined_models_to_dense(Int_model_sp,Int_model_tp,RAT_DATA_sp,RAT_DATA_tp,Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model)
    
    for i in range(15):
        K.clear_session()
        
    print(Ratnum + ': Fold ' + str(foldnum) + ' done')
    
    return None

def CNN_CM_CM_DD(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model):

    ''' Metric functions'''
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall))
    
    """
    CNN code
    """
    
    RAT_data = None
    
    RAT_data = PP.Preprocessing_module(Ratnum,foldnum)
    RAT_data.Randomize_Train_and_getValidSet(RAT_data.training_set,RAT_data.training_labels,numspikes)
    
    """
    Labels and Training Reformating
    """
    
    labels_training = None
    labels_test = None
    labels_valid = None
    
    
    labels_training = tf.keras.utils.to_categorical(RAT_data.training_labels - 1, None) #labels_training = keras.utils.to_categorical(RAT_data.training_labels - 1, None)
    labels_test = tf.keras.utils.to_categorical(RAT_data.test_labels - 1, None) #labels_test = keras.utils.to_categorical(RAT_data.test_labels - 1, None)
    labels_valid = tf.keras.utils.to_categorical(RAT_data.valid_labels - 1, None) #labels_valid = keras.utils.to_categorical(RAT_data.valid_labels - 1, None)
    
    input_img = Input(shape=(RAT_data.numcons,100,1)) #input_img = Input(shape=(56,100,1))
    Conv1_out = Conv2D(numfilters, (filtsize, filtsize), activation='relu', padding='same', name = "conv1")(input_img)
    MPool1 = MaxPooling2D((4, 4), padding='same', name = "mpool1")(Conv1_out)
    
    temp_out = Flatten()(MPool1)
    temp_dense1 = Dense(dense_neurons*4, activation='relu')(temp_out)
    temp_dense2 = Dense(3, activation='softmax')(temp_dense1)
    
    Initial_guess1 = Model(input_img,temp_dense2)
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Initial_guess1.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    if print_model:
        print(Initial_guess1.summary())
    # else:
    Initial_guess1.fit(RAT_data.training_set, labels_training,
              epochs=25,batch_size = size_batch, shuffle=True,
              validation_data=(RAT_data.valid_set, labels_valid))
    
    Max_pool1 = MaxPooling2D((2, 2), padding='same', name = "max_pool1")(Conv1_out)
    filtsize1 = int(filtsize / 2) + (filtsize % 2 > 0)
    Conv2_out = Conv2D(numfilters*channel_width_multiplier, (filtsize1, filtsize1), activation='relu', padding='same', name = "conv2")(Max_pool1)
    MPool2 = MaxPooling2D((4, 4), padding='same', name = "mpool2")(Conv2_out)
    
    temp_out2 = Flatten()(MPool2)
    temp_dense11 = Dense(dense_neurons*2, activation='relu')(temp_out2)
    temp_dense22 = Dense(3, activation='softmax')(temp_dense11)
    
    Initial_guess2 = Model(input_img,temp_dense22)
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Initial_guess2.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    if print_model:
        print(Initial_guess2.summary())
    
    Initial_guess2.fit(RAT_data.training_set, labels_training,
              epochs=25,batch_size = size_batch, shuffle=True,
              validation_data=(RAT_data.valid_set, labels_valid))
    
    Max_pool2 = MaxPooling2D((2, 2), padding='same', name = "max_pool2")(Conv2_out)
    filtsize2 = int(filtsize / 4) + (filtsize % 4 > 0)
    Conv3_out = Conv2D(2*numfilters*(channel_width_multiplier**2), (filtsize2, filtsize2), activation='relu', padding='same', name = "conv3")(Max_pool2)
    
    conv1x1 = Conv2D(numfilters/2, (1, 1), activation='relu', 
                   name = "conv1x1")(Conv3_out)
    
    # conv1x1 = Conv2D(numfilters/4, (1, 1), activation='relu', 
    #                name = "conv1x1-2")(conv1x1)
    
    out = Flatten()(conv1x1)
    dense1 = Dense(dense_neurons, activation='relu', name = "Den1")(out)
    Dropout1 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(3, activation='softmax', name = "Den2")(Dropout1)
    
    Full_model = Model(input_img,dense2)
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Full_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    early_stopping_monitor = EarlyStopping(patience=valid_patience,
                                           restore_best_weights=(True))
    
    Intermediate_model = Model(input_img,Full_model.get_layer("conv1x1").output)
    
    if print_model:
        print(Full_model.summary())
        # return [Intermediate_model, RAT_data]
    
    history = Full_model.fit(RAT_data.training_set, labels_training,
                  epochs=numepochs,batch_size = size_batch, shuffle=True,
                  validation_data=(RAT_data.valid_set, labels_valid), callbacks=[early_stopping_monitor])
          
#    model.fit(training_data, labels_training, shuffle = True,
#              epochs=numepochs,batch_size = size_batch)
    
    score = Full_model.evaluate(RAT_data.test_set, labels_test)
    np.disp(score)
    class_probs = Full_model.predict(RAT_data.test_set)
    
    folder = rat_folder_main
    filename_prefix = filename_prefix_main
    
    File_utils.save_files(folder,filename_prefix,class_probs,RAT_data.test_labels,history)
    
    
    del Full_model
    
    return [Intermediate_model,RAT_data]


def CNN_CM_CM_DD_ConPerRing(Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,samples1,samples2,samples3,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model):

    ''' Metric functions'''
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall))
    
    """
    CNN code
    """
    
    RAT_data = None
    
    RAT_data = PP2.Preprocessing_module(Ratnum,foldnum)
    RAT_data.Reshape_data_set(RAT_data.training_set,RAT_data.test_set)
    RAT_data.Randomize_Train_and_getValidSet2(RAT_data.training_set,RAT_data.training_labels,samples1,samples2,
                                         samples3,numspikes)
    
    """
    Labels and Training Reformating
    """
    labels_training = None
    labels_test = None
    labels_valid = None
    
    
    labels_training = tf.keras.utils.to_categorical(RAT_data.training_labels - 1, None) #labels_training = keras.utils.to_categorical(RAT_data.training_labels - 1, None)
    labels_test = tf.keras.utils.to_categorical(RAT_data.test_labels - 1, None) #labels_test = keras.utils.to_categorical(RAT_data.test_labels - 1, None)
    labels_valid = tf.keras.utils.to_categorical(RAT_data.valid_labels - 1, None) #    labels_valid = keras.utils.to_categorical(RAT_data.valid_labels - 1, None)
   
    
    input_img = Input(shape=(RAT_data.numcons,100,1)) #input_img = Input(shape=(56,100,1))
    Conv1_out = Conv2D(numfilters, (filtsize, filtsize), activation='relu', padding='same', name = "conv1")(input_img)
    MPool1 = MaxPooling2D((4, 4), padding='same', name = "mpool1")(Conv1_out)
    
    temp_out = Flatten()(MPool1)
    temp_dense1 = Dense(dense_neurons*4, activation='relu')(temp_out)
    temp_dense2 = Dense(3, activation='softmax')(temp_dense1)
    
    Initial_guess1 = Model(input_img,temp_dense2)
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Initial_guess1.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    if print_model:
        print(Initial_guess1.summary())
    
    Initial_guess1.fit(RAT_data.training_set, labels_training,
              epochs=25,batch_size = size_batch, shuffle=True,
              validation_data=(RAT_data.valid_set, labels_valid))
    
    Max_pool1 = MaxPooling2D((2, 2), padding='same', name = "max_pool1")(Conv1_out)
    filtsize1 = int(filtsize / 2) + (filtsize % 2 > 0)
    Conv2_out = Conv2D(numfilters*channel_width_multiplier, (filtsize1, filtsize1), activation='relu', padding='same', name = "conv2")(Max_pool1)
    MPool2 = MaxPooling2D((4, 4), padding='same', name = "mpool2")(Conv2_out)
    
    temp_out2 = Flatten()(MPool2)
    temp_dense11 = Dense(dense_neurons*2, activation='relu')(temp_out2)
    temp_dense22 = Dense(3, activation='softmax')(temp_dense11)
    
    Initial_guess2 = Model(input_img,temp_dense22)
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Initial_guess2.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    if print_model:
        print(Initial_guess2.summary())
    
    Initial_guess2.fit(RAT_data.training_set, labels_training,
              epochs=25,batch_size = size_batch, shuffle=True,
              validation_data=(RAT_data.valid_set, labels_valid))
    
    Max_pool2 = MaxPooling2D((2, 2), padding='same', name = "max_pool2")(Conv2_out)
    filtsize2 = int(filtsize / 4) + (filtsize % 4 > 0)
    Conv3_out = Conv2D(2*numfilters*(channel_width_multiplier**2), (filtsize2, filtsize2), activation='relu', padding='same', name = "conv3")(Max_pool2)
    
    conv1x1 = Conv2D(numfilters/2, (1, 1), activation='relu', 
                   name = "conv1x1")(Conv3_out)
    
    # conv1x1 = Conv2D(numfilters/4, (1, 1), activation='relu', 
    #                name = "conv1x1-2")(conv1x1)
    
    out = Flatten()(conv1x1)
    dense1 = Dense(dense_neurons, activation='relu', name = "Den1")(out)
    Dropout1 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(3, activation='softmax', name = "Den2")(Dropout1)
    
    Full_model = Model(input_img,dense2)
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Full_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    early_stopping_monitor = EarlyStopping(patience=valid_patience,
                                           restore_best_weights=(True))
    
    Intermediate_model = Model(input_img,Full_model.get_layer("conv1x1").output)
    
    if print_model:
        print(Full_model.summary())
        # return [Intermediate_model, RAT_data]
    
    history = Full_model.fit(RAT_data.training_set, labels_training,
          epochs=numepochs,batch_size = size_batch, shuffle=True,
          validation_data=(RAT_data.valid_set, labels_valid), callbacks=[early_stopping_monitor])
          
#    model.fit(training_data, labels_training, shuffle = True,
#              epochs=numepochs,batch_size = size_batch)
    
    score = Full_model.evaluate(RAT_data.test_set, labels_test)
    np.disp(score)
    class_probs = Full_model.predict(RAT_data.test_set)
    
    folder = rat_folder_ConPerRing_main
    filename_prefix = filename_prefix_ConPerRing_main
    
    File_utils.save_files(folder,filename_prefix,class_probs,RAT_data.test_labels,history)
    

    del Full_model
    
    return [Intermediate_model,RAT_data]

def combined_models_to_dense(Int_model1,Int_model2,RAT_data,RAT_data2,Ratnum,foldnum,numepochs,size_batch,valid_patience,numspikes,numfilters,filtsize, dropout_rate,dense_neurons,channel_width_multiplier,print_model):
    
    ''' Metric functions'''
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall))
    
    
    input1_img = Input(shape=(RAT_data.numcons,100,1)) #input1_img = Input(shape=(56,100,1))
    input2_img = Input(shape=(RAT_data.numcons,100,1)) #input2_img = Input(shape=(56,100,1))
    
    Conv11_out = Conv2D(numfilters, (filtsize, filtsize), activation='relu', padding='same', name = "conv11")(input1_img)
    Conv21_out = Conv2D(numfilters, (filtsize, filtsize), activation='relu', padding='same', name = "conv21")(input2_img)
    
    
    MPool11_out = MaxPooling2D((2, 2), padding='same', name = "MP11")(Conv11_out)
    MPool21_out = MaxPooling2D((2, 2), padding='same', name = "MP21")(Conv21_out)
    
    filtsize1 = int(filtsize / 2) + (filtsize % 2 > 0)
    Conv12_out = Conv2D(numfilters*channel_width_multiplier, (filtsize1, filtsize1), activation='relu', padding='same', name = "conv12")(MPool11_out)
    Conv22_out = Conv2D(numfilters*channel_width_multiplier, (filtsize1, filtsize1), activation='relu', padding='same', name = "conv22")(MPool21_out)
    
    
    MPool12_out = MaxPooling2D((2, 2), padding='same', name = "MP12")(Conv12_out)
    MPool22_out = MaxPooling2D((2, 2), padding='same', name = "MP22")(Conv22_out)
    
    filtsize2 = int(filtsize / 4) + (filtsize % 4 > 0)
    Conv13_out = Conv2D(2*numfilters*(channel_width_multiplier**2), (filtsize2, filtsize2), activation='relu', padding='same', name = "conv13")(MPool12_out)
    Conv23_out = Conv2D(2*numfilters*(channel_width_multiplier**2), (filtsize2, filtsize2), activation='relu', padding='same', name = "conv23")(MPool22_out)    
    
    conv1x1_1 = Conv2D(numfilters/2, (1, 1), activation='relu', 
                   name = "conv1x1_1")(Conv13_out)
    conv1x1_2 = Conv2D(numfilters/2, (1, 1), activation='relu', 
                   name = "conv1x1_2")(Conv23_out)
    
    # conv1x1_1 = Conv2D(numfilters/4, (1, 1), activation='relu', 
    #                name = "conv1x1-2_1")(conv1x1_1)
    # conv1x1_2 = Conv2D(numfilters/4, (1, 1), activation='relu', 
    #                name = "conv1x1-2_2")(conv1x1_2)
        
    Merged_input_bf_DENSE = keras.layers.concatenate([conv1x1_1, conv1x1_2])   
    
    Flat_out = Flatten()(Merged_input_bf_DENSE)
    Dense1 = Dense(dense_neurons*2, activation='relu')(Flat_out)
    Dropout1 = Dropout(dropout_rate)(Dense1)
    Dense2 = Dense(3, activation='softmax')(Dropout1)    
    
    Combined_model = Model(inputs=[input1_img, input2_img], outputs = Dense2)
        
    C11_w = Int_model1.get_layer("conv1").get_weights()
    C21_w = Int_model2.get_layer("conv1").get_weights()
    
    M11_w = Int_model1.get_layer("max_pool1").get_weights()
    M21_w = Int_model2.get_layer("max_pool1").get_weights()
    
    C12_w = Int_model1.get_layer("conv2").get_weights()
    C22_w = Int_model2.get_layer("conv2").get_weights()
    
    M12_w = Int_model1.get_layer("max_pool2").get_weights()
    M22_w = Int_model2.get_layer("max_pool2").get_weights()
    
    C13_w = Int_model1.get_layer("conv3").get_weights()
    C23_w = Int_model2.get_layer("conv3").get_weights()   
    
    C1x1_1_w = Int_model1.get_layer("conv1x1").get_weights()
    C1x1_2_w = Int_model2.get_layer("conv1x1").get_weights()  
    
    # C1x1_2_1_w = Int_model1.get_layer("conv1x1-2").get_weights()
    # C1x1_2_2_w = Int_model2.get_layer("conv1x1-2").get_weights()  
    
    Combined_model.get_layer("conv11").set_weights(C11_w)
    Combined_model.get_layer("conv21").set_weights(C21_w)
    
    Combined_model.get_layer("MP11").set_weights(M11_w)
    Combined_model.get_layer("MP21").set_weights(M21_w)
    
    Combined_model.get_layer("conv12").set_weights(C12_w)
    Combined_model.get_layer("conv22").set_weights(C22_w)
    
    Combined_model.get_layer("MP12").set_weights(M12_w)
    Combined_model.get_layer("MP22").set_weights(M22_w)
    
    Combined_model.get_layer("conv13").set_weights(C13_w)
    Combined_model.get_layer("conv23").set_weights(C23_w)  
    
    Combined_model.get_layer("conv1x1_1").set_weights(C1x1_1_w)
    Combined_model.get_layer("conv1x1_2").set_weights(C1x1_2_w)  
    
    # Combined_model.get_layer("conv1x1-2_1").set_weights(C1x1_2_1_w)
    # Combined_model.get_layer("conv1x1-2_2").set_weights(C1x1_2_2_w)  
    
#    Combined_model.get_layer("conv11").trainable = False
#    Combined_model.get_layer("conv21").trainable = False
#    
#    Combined_model.get_layer("MP11").trainable = False
#    Combined_model.get_layer("MP21").trainable = False
#    
#    Combined_model.get_layer("conv12").trainable = False
#    Combined_model.get_layer("conv22").trainable = False
#    
#    Combined_model.get_layer("MP12").trainable = False
#    Combined_model.get_layer("MP22").trainable = False
#    
#    Combined_model.get_layer("conv13").trainable = False
#    Combined_model.get_layer("conv23").trainable = False  
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    Combined_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy',f1],)
    
    early_stopping_monitor = EarlyStopping(patience=valid_patience,
                                           restore_best_weights=(True))
    
    labels_training = None
    labels_test = None
    labels_valid = None
    
    
    labels_training = tf.keras.utils.to_categorical(RAT_data.training_labels - 1, None) #labels_training = keras.utils.to_categorical(RAT_data.training_labels - 1, None)
    labels_test = tf.keras.utils.to_categorical(RAT_data.test_labels - 1, None) #labels_test = keras.utils.to_categorical(RAT_data.test_labels - 1, None)
    labels_valid = tf.keras.utils.to_categorical(RAT_data.valid_labels - 1, None) #labels_valid = keras.utils.to_categorical(RAT_data.valid_labels - 1, None)
    
    if print_model:
        print(Combined_model.summary())
        # return None
    
    history = Combined_model.fit([RAT_data.training_set, RAT_data2.training_set], labels_training,
              epochs=1000,batch_size = size_batch, shuffle=True,
              validation_data=([RAT_data.valid_set, RAT_data2.valid_set], labels_valid), callbacks=[early_stopping_monitor])
    
    score = Combined_model.evaluate([RAT_data.test_set, RAT_data2.test_set], labels_test)
    np.disp(score)
    class_probs = Combined_model.predict([RAT_data.test_set, RAT_data2.test_set])
    
    folder = rat_folder_combined_main
    filename_prefix = filename_prefix_combined_main
    
    File_utils.save_files(folder,filename_prefix,class_probs,RAT_data.test_labels,history)
      
    #test 
    del Combined_model
    
    K.clear_session()
    
    return None
    
    
