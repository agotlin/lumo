# Purpose: This script contains helper functions for the SpeedPrediction_ Jupyter Notebook script.

# Import all necessary libraries

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Input, Lambda, Merge
from keras.layers.normalization import BatchNormalization 
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras import regularizers 
from keras import optimizers
from sklearn.model_selection import GridSearchCV 
from keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import tensorflow as tf

from datetime import datetime
from dateutil import tz
from IPython import embed
import time
from time import strftime, gmtime
import socket
import pickle
import os.path
#import dill #can't find dill

from pathlib import Path
from sklearn.metrics import confusion_matrix 

np.random.seed(7) # Set seed for reproducibility

# Helper Functions

def read_data(file_path):
    data = pd.read_csv(file_path,header = 0) # This uses the header row (row 0) as the column names
    return data

def windows(data, size): # define time windows to create each training example
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += sample_stride # other common options: (size / 2)
    
# Used for vectorized input WITH SQL pre-processing
def segment_signal_FCN_vector(data_inputs, data_full): 
    #dataframe_input = data_inputs.loc[:, 'gender':'pelvic_tilt_lag_0'] # select all columns from gender to pelvic_tilt_lag_0
    dataframe_input = data_inputs.loc[:, 'gender':'pelvic_rotation_lag_0'] # select all columns from gender to pelvic_rotation_lag_0
    dataframe_labels = data_full.loc[:, 'gps_speed_lag_7':'gps_speed_lag_3'] # select all columns from gender to pelvic_tilt_lag_0
    segments = dataframe_input.values
    labels_before_avg = dataframe_labels.values
    if speed_bucket_size == '0.1':
        labels = np.around(np.mean(labels_before_avg, axis=1),decimals=1)
    elif speed_bucket_size == 'none_use_regression':
        labels = np.mean(labels_before_avg, axis=1)
    return segments, labels

# Used for CNN/FFCN input WITHOUT SQL pre-processing
def segment_signal_w_concat(data_inputs, data_full, input_window_size = input_window_size):
    # define segment shape for training example input
    if  model_architecture == 'FCN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))
    elif model_architecture == 'CNN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))    
        segments_timeseries = np.empty((0, input_window_size, num_channels))
        segments_anthro = np.empty((0, num_anthropometrics))
    labels = np.empty((0))
    for (start, end) in windows(data_full['timestamp'], input_window_size):
        a = data_inputs["bounce"][start:end]
        b = data_inputs["braking"][start:end]
        c = data_inputs["cadence"][start:end]
        d = data_inputs["ground_contact"][start:end]
        e = data_inputs["pelvic_drop"][start:end]
        f = data_inputs["pelvic_rotation"][start:end]
        aa = data_inputs["age"][start]
        bb = data_inputs["weight"][start]
        cc = data_inputs["height"][start]
        dd = data_inputs["gender"][start]   
        if(end < data_full.shape[0] and len(data_full['timestamp'][start:end]) == input_window_size and data_full['activity_id'][start]==data_full['activity_id'][end]):
            # Create segment input arrays
            if  model_architecture == 'FCN':
                segments_toadd = np.vstack([np.dstack([a,b,c,d,e,f])])
                segments_toadd_reshape = segments_toadd.reshape(input_window_size * num_channels)
                segments = np.vstack([segments,np.hstack([aa,bb,cc,dd,segments_toadd_reshape])])
            elif model_architecture == 'CNN':        
                segments_timeseries = np.vstack([segments_timeseries,np.dstack([a,b,c,d,e,f])])
                segments_anthro = np.vstack([segments_anthro,np.hstack([aa,bb,cc,dd])])
            # Create labels array
            start_labeling = np.int(np.floor(start+(end-start)/2) - np.floor(label_window_size/2))
            end_labeling = start_labeling + label_window_size
            if speed_bucket_size == '0.1':
                labels = np.append(labels,np.around(np.mean(data_full["gps_speed_true"][start_labeling:end_labeling]),decimals=1)) # round to nearest decimal
            elif speed_bucket_size == 'none_use_regression':
                labels = np.append(labels,np.mean(data_full["gps_speed_true"][start_labeling:end_labeling])) # no rounding, use regression
    if  model_architecture == 'FCN':
        return segments, labels
    elif model_architecture == 'CNN':            
        return segments_timeseries, segments_anthro, labels

def load_results_file_FCN(results_file_name):
    my_file = Path(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv")
    if my_file.is_file():
        print("Found results file")
        prev_results=pd.read_csv(my_file,header=0)
        print(list(prev_results.columns.values))
        return prev_results
    else:
        print("no results file found - creating file")
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            batch_size,
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            "",
            "",
            "",
            num_hidden_fc_layers,
            hidden_units_strategy,
            activations_strategy,
            dropout_rates
            ]]
        
        df=pd.DataFrame(a, columns=["model type",
                                    "model filename",
                                    "plot filename",
                                    "data filename",
                                    "epochs",
                                    "runtime",
                                    "train accuracy 1",
                                    "dev accuracy 1",
                                    "train accuracy 2",
                                    "dev accuracy 2",
                                    "train accuracy 3",
                                    "dev accuracy 3",
                                    "batch_size",
                                    "learning_rate",
                                    "speed_bucket_size",
                                    "loss_function",
                                    "input_window_size",
                                    "label_window_size",
                                    "optimizer_type",
                                    "evaluation_metric_1",
                                    "evaluation_metric_2",
                                    "evaluation_metric_3",
                                    "num_hidden_fc_layers",
                                    "hidden_units_strategy",
                                    "activations_strategy",
                                    "dropout_rates"])
        
        df.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        return df

def load_results_file_CNN(results_file_name):
    my_file = Path(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv")
    if my_file.is_file():
        #print("Found results file")
        prev_results=pd.read_csv(my_file,header=0)
        #print(list(prev_results.columns.values))
        return prev_results
    else:
        print("no results file found - creating file")
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            batch_size,
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            "",
            "",
            "",
            hidden_units_strategy_CNN,
            num_filters,
            kernel_size,
            sample_stride,
            activation_conv_layer,
            activations_strategy_CNN,
            max_pool_kernel_size
            ]]
        
        df=pd.DataFrame(a, columns=["model type",
                                    "model filename",
                                    "plot filename",
                                    "data filename",
                                    "epochs",
                                    "runtime",
                                    "dev accuracy 1",
                                    "train accuracy 1",
                                    "dev accuracy 2",
                                    "train accuracy 2",
                                    "dev accuracy 3",
                                    "train accuracy 2",
                                    "batch_size",
                                    "learning_rate",
                                    "speed_bucket_size",
                                    "loss_function",
                                    "input_window_size",
                                    "label_window_size",
                                    "optimizer_type",
                                    "evaluation_metric_1",
                                    "evaluation_metric_2",
                                    "evaluation_metric_3",
                                    "hidden_units_strategy_CNN",
                                    "num_filters",
                                    "kernel_size",
                                    "sample_stride",
                                    "activation_conv_layer",
                                    "activations_strategy_CNN",
                                    "max_pool_kernel_size"])
        
        df.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        return df   