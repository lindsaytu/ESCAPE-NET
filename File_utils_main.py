# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 2021

@author: hwangyic
"""

import scipy.io
import matplotlib.pyplot as plt
import os
#import keras
import tensorflow
import numpy as np
from tensorflow.keras import backend as K
from os import path
from enum import Enum
from tensorflow.keras.layers import Layer


# from tensorflow.python.saved_model import loader_impl
# from tensorflow.python.keras.saving.saved_model import load as saved_model_load

 # Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class Utils:
    '''Constructor'''    
    def __init__(self,root_dir):
        self.root_dir = root_dir
        
    def get_root_dir(self):
        return self.root_dir 
    
    from main import data_files
    data_files()
        
    class LossFunction(Enum):
         CATEGORICAL_CROSSENTROPY = 1
         MACRO_SOFT_F1 = 2
         CCE_MSF1 = 3
         
    class LayerType(Enum):
         LSTM = 1
         GRU = 2
         BIDIR_LSTM = 3
         BIDIR_GRU = 4
         
    class Activation(Enum):
         ReLU = 1 
         TANH = 2
         
    class OptModelType(Enum):
        TF_LITE = 1
         
   
    ''' Metric functions'''
    @staticmethod
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
            recall = float(true_positives) / (float(possible_positives) + 1e-07) # 1e-07 = K.epsilon, fuzz factor, as of Feb 2021
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
            precision = float(true_positives) / (float(predicted_positives) + 1e-07)
            return precision
        y_true = np.eye(3)[y_true-1]
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        # print("precision: " + str(precision) + "\n")
        # print("recall: " + str(recall) + "\n")
        return 2*((precision*recall)/(precision+recall))
    
    @staticmethod
    def f1_macro(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
    
            Only computes a batch-wise average of recall.
    
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)),axis=0)
            possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)),axis=0)
            recall = true_positives.astype(np.float) / (possible_positives.astype(np.float) + 1e-07) # 1e-07 = K.epsilon, fuzz factor, as of Feb 2021
            return recall
    
        def precision(y_true, y_pred):
            """Precision metric.
    
            Only computes a batch-wise average of precision.
    
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)),axis=0)
            predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)),axis=0)
            precision = true_positives.astype(np.float) / (predicted_positives.astype(np.float) + 1e-07)
            return precision
        y_true = np.eye(3)[y_true-1]
        
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall + 1e-07))
    
    @staticmethod
    def macro_soft_f1(y, y_hat):
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.
        
        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        
        y = tensorflow.cast(y, tensorflow.float32)
        y_hat = tensorflow.cast(y_hat, tensorflow.float32)
        tp = tensorflow.reduce_sum(y_hat * y, axis=0)
        fp = tensorflow.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tensorflow.reduce_sum((1 - y_hat) * y, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tensorflow.reduce_mean(cost) # average on all labels
        
        return macro_cost
    
    
    @staticmethod
    def cce_msf1(y_true, y_pred):
        cce = tensorflow.keras.losses.CategoricalCrossentropy()
        combo_loss = cce(y_true, y_pred) + Utils.macro_soft_f1(y_true, y_pred)
        return combo_loss/2
    
    @staticmethod
    def f1_tf(y, y_hat, thresh=0.5):
        """Compute the F1-score on a batch of observations (F1 across all examples)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            f1 (scalar): value of F1 for the batch
        """
        y_pred = tensorflow.cast(tensorflow.greater(y_hat, thresh), tensorflow.float32)
        tp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * y), tensorflow.float32)
        fp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * (1 - y)), tensorflow.float32)
        fn = tensorflow.cast(tensorflow.math.count_nonzero((1 - y_pred) * y), tensorflow.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        
        
        return f1
    
    @staticmethod
    def macro_f1(y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tensorflow.cast(tensorflow.greater(y_hat, thresh), tensorflow.float32)
        tp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * y, axis=0), tensorflow.float32)
        fp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * (1 - y), axis=0), tensorflow.float32)
        fn = tensorflow.cast(tensorflow.math.count_nonzero((1 - y_pred) * y, axis=0), tensorflow.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tensorflow.reduce_mean(f1)
        
        
        return macro_f1
    
    @staticmethod
    def f1_c0(y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tensorflow.cast(tensorflow.greater(y_hat, thresh), tensorflow.float32)
        tp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * y, axis=0), tensorflow.float32)
        fp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * (1 - y), axis=0), tensorflow.float32)
        fn = tensorflow.cast(tensorflow.math.count_nonzero((1 - y_pred) * y, axis=0), tensorflow.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        
        return f1[0]
    @staticmethod
    def f1_c1(y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tensorflow.cast(tensorflow.greater(y_hat, thresh), tensorflow.float32)
        tp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * y, axis=0), tensorflow.float32)
        fp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * (1 - y), axis=0), tensorflow.float32)
        fn = tensorflow.cast(tensorflow.math.count_nonzero((1 - y_pred) * y, axis=0), tensorflow.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        
        return f1[1]
    
    @staticmethod
    def f1_c2(y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)
        
        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive
            
        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tensorflow.cast(tensorflow.greater(y_hat, thresh), tensorflow.float32)
        tp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * y, axis=0), tensorflow.float32)
        fp = tensorflow.cast(tensorflow.math.count_nonzero(y_pred * (1 - y), axis=0), tensorflow.float32)
        fn = tensorflow.cast(tensorflow.math.count_nonzero((1 - y_pred) * y, axis=0), tensorflow.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        
        return f1[2]
    
   
    
    
    dependencies = {
             'f1': f1.__func__,
             'f1_tf': f1_tf.__func__,
             'macro_soft_f1': macro_soft_f1.__func__,
             'macro_f1': macro_f1.__func__,
             'f1_c0': f1_c0.__func__,
             'f1_c1': f1_c1.__func__,
             'f1_c2': f1_c2.__func__,
             'attention': attention
        }
    
    
    def hello_world(self):
        print("hello world")
        print(self.root_dir)
    
    
    def load_model(self,folder,filename_prefix,dependencies,int_model=False):#,root_dir='\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts\\'):
        directory = self.root_dir + folder    
        full_filename_prefix = directory + filename_prefix
        
        model_suffix = ''
        
        if int_model:
            model_suffix = '_only_conv_int_model.h5'
        else:
            model_suffix = '_only_conv_model.h5'
            
        full_filename = full_filename_prefix + model_suffix
        if not path.exists(full_filename):
            print(full_filename)
            print("debEug NOT returning a file")
            return None
        
        model = tensorflow.keras.models.load_model(full_filename, custom_objects=dependencies, compile = False)
        return model
    
    def load_model_args(self,folder,filename_prefix,dependencies,int_model=False):#,root_dir='\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts\\'):
        directory = self.root_dir + folder    
        full_filename_prefix = directory + filename_prefix
        
        model_suffix = ''
        
        if int_model:
            model_suffix = '_only_conv_int_model.h5'
        else:
            model_suffix = '_only_conv_model.h5'
            
        full_filename = full_filename_prefix + model_suffix
        if not path.exists(full_filename):
            # print(full_filename)
            # print("debEug NOT returning a file")
            return None
        
        # model = keras.models.load_model(full_filename, custom_objects=dependencies)
        return full_filename, dependencies
    
    def save_model(self,folder,filename_prefix,model):#,root_dir='\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts\\'):
        directory = self.root_dir + folder
        try:  
            os.makedirs(directory, exist_ok=True) 
        except FileExistsError:
            pass
        
        full_filename_prefix = directory + filename_prefix
        
        model.save(full_filename_prefix + '_only_conv_model.h5')
    
    def save_files(self,folder,filename_prefix,class_probs=[],test_labels=[],history=None,
                   Full_model=None,Intermediate_model=None,mat_filename=''):
                   #root_dir='\\\\svm_uhn.uhn.ca\\NET\\NET2\\Zariffa\\Eugene\\NN Scripts\\'):
        directory = self.root_dir + folder
        try:  
            os.makedirs(directory, exist_ok=True) 
        except FileExistsError:
            pass
        
        class_probs = np.asarray(class_probs)
        test_labels = np.asarray(test_labels)
        
        full_filename_prefix = directory + filename_prefix
        if class_probs.size != 0 and test_labels.size != 0:
            if mat_filename == '':
                mat_filename = full_filename_prefix + '_only_conv.mat'
            print("DEBEUG SAVING CLASS_PROBS TO: ")
            print(mat_filename)
            scipy.io.savemat(mat_filename,{'class_probs':class_probs,'test_labels':test_labels})
        if Full_model != None:
            print("DEBEUG SAVING Full_model TO: ")
            print(full_filename_prefix + '_only_conv_model.h5')
            Full_model.save(full_filename_prefix + '_only_conv_model.h5')
        if Intermediate_model!= None:
            print("DEBEUG SAVING Intermediate_model TO: ")
            print(full_filename_prefix + '_only_conv_int_model.h5')
            Intermediate_model.save((full_filename_prefix + '_only_conv_int_model.h5'))
        
        if history != None:
            plt.plot(history.history['loss'], label='categorical_crossentropy (training data)')
            plt.plot(history.history['val_loss'], label='categorical_crossentropy (validation data)')
            plt.title('Categorical cross entropy')
            plt.ylabel('Categorical cross entropy value')
            plt.xlabel('No. epoch')
            plt.legend(loc="upper left")
            fig = plt.gcf()
            fig.savefig(full_filename_prefix + '_CCE.png')
            #plt.show()
            plt.clf()
