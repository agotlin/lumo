# Purpose: This script contains helper functions for the SpeedPrediction_ Jupyter Notebook script.

# NOTE: There are two version of this file, the user should keep this version in sync: directories [lumo\Codebase\] and [lumo\Codebase\PyFilesForSherlock]

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

from keras import backend as K

from matplotlib.patches import Rectangle
from matplotlib.legend import Legend

np.random.seed(7) # Set seed for reproducibility


# Helper Functions


# Read and process input data

def read_data(file_path):
    data = pd.read_csv(file_path,header = 0) # This uses the header row (row 0) as the column names
    return data

def windows(data, size, sample_stride): # define time windows to create each training example
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
def segment_signal_w_concat(data_inputs, data_full, model_architecture, speed_bucket_size, input_window_size, num_channels, num_anthropometrics, label_window_size, sample_stride, speed_minimum): 
    # define segment shape for training example input
    if  model_architecture == 'FCN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))
    elif model_architecture == 'CNN':
        segments = np.empty((0,input_window_size*num_channels + num_anthropometrics))    
        segments_timeseries = np.empty((0, input_window_size, num_channels))
        segments_anthro = np.empty((0, num_anthropometrics))
    labels = np.empty((0))
    for (start, end) in windows(data_full['timestamp'], input_window_size, sample_stride):
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
        start_labeling = np.int(np.floor(start+(end-start)/2) - np.floor(label_window_size/2))
        end_labeling = start_labeling + label_window_size     
        if(end < data_full.shape[0] and 
            len(data_full['timestamp'][start:end]) == input_window_size and 
            data_full['activity_id'][start]==data_full['activity_id'][end] and 
            not np.any(data_full['precededbynulls'][start:end] == 1) and
            np.mean(data_full["gps_speed_true"][start_labeling:end_labeling]) > speed_minimum): # speed must be greater than speed_minimum to be considered as a training example
            # Create segment input arrays
            if  model_architecture == 'FCN':
                segments_toadd = np.vstack([np.dstack([a,b,c,d,e,f])])
                segments_toadd_reshape = segments_toadd.reshape(input_window_size * num_channels)
                segments = np.vstack([segments,np.hstack([aa,bb,cc,dd,segments_toadd_reshape])])
            elif model_architecture == 'CNN':        
                segments_timeseries = np.vstack([segments_timeseries,np.dstack([a,b,c,d,e,f])])
                segments_anthro = np.vstack([segments_anthro,np.hstack([aa,bb,cc,dd])])
            # Create labels array
#             start_labeling = np.int(np.floor(start+(end-start)/2) - np.floor(label_window_size/2))
#             end_labeling = start_labeling + label_window_size
            if speed_bucket_size == '0.1':
                labels = np.append(labels,np.around(np.mean(data_full["gps_speed_true"][start_labeling:end_labeling]),decimals=1)) # round to nearest decimal
            elif speed_bucket_size == 'none_use_regression':
                labels = np.append(labels,np.mean(data_full["gps_speed_true"][start_labeling:end_labeling])) # no rounding, use regression
    if  model_architecture == 'FCN':
        return segments, labels
    elif model_architecture == 'CNN':            
        return segments_timeseries, segments_anthro, labels
    

    
# Create learning curves

def create_learning_curves_from_model(machine_to_run_script
            ,trainAccuracy_4
            ,devAccuracy_4
            ,trainAccuracy_1
            ,devAccuracy_1
            ,trainAccuracy_2
            ,devAccuracy_2
            ,dev_reporting_metric_1
            ,final_accuracy_dev_1
            ,dev_reporting_metric_2
            ,final_accuracy_dev_2
            ,loss_function
            ,learning_rate
            ,batch_size
            ,speed_bucket_size
            ,training_epochs
            ,input_window_size
            ,labels_to_number
            ,accuracy_reporting_metric_1
            ,accuracy_reporting_metric_2
            ,accuracy_reporting_metric_3
            ,plot_note
            ,folder_head_loc
            ,file_name):
    if machine_to_run_script == 'local':
        fig, ax1 = plt.subplots()
        lines=[]

        lines += ax1.plot(trainAccuracy_1,'#FF5733', label='Train Accuracy 1', linewidth=1)
        lines += ax1.plot(devAccuracy_1,'#C70039', label='Dev Accuracy 1', linewidth=1)
        lines += ax1.plot(trainAccuracy_2,'#9C27B0', label='Train Accuracy 2', linewidth=1)
        lines += ax1.plot(devAccuracy_2,'#7986CB', label='Dev Accuracy 2', linewidth=1)
        if speed_bucket_size != 'none_use_regression':
            lines += ax1.plot(trainAccuracy_4,'#0e128c', label='Train Accuracy 4', linewidth=1) #'#DAF7A6'
            lines += ax1.plot(devAccuracy_4,'#a3a4cc', label='Dev Accuracy 4', linewidth=1)# '#33FF00',
            plt.ylim([0.0, 1.0])  # Surpress this for classification tasks

        plt.ylabel('Train vs. Dev Accuracy')

        plt.xlabel('Epochs')
        plt.title(dev_reporting_metric_1 + ": " + str(np.around(final_accuracy_dev_1,4)) + "\n" + \
                 dev_reporting_metric_2 + ": " + str(np.around(final_accuracy_dev_2,4))) 
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        plt.legend([extra,extra,extra,extra,extra,extra,extra,extra,extra,extra],(
                                                        "loss: " + loss_function,
                                                        "learning rate: " + str(learning_rate),
                                                        "batch_size: " + str(batch_size),
                                                        "speed_bucket_size: " + speed_bucket_size,
                                                        "epochs: "+str(training_epochs),
                                                        "input_window_size: " + str(input_window_size),
                                                        "num_labels: " + str(len(labels_to_number)),
                                                        "evaluation metric 1:"+accuracy_reporting_metric_1,
                                                        "evaluation metric 2:"+accuracy_reporting_metric_2,
                                                        "evaluation metric 3:"+accuracy_reporting_metric_3,
                                                        "note:" + plot_note),
                                                        bbox_to_anchor=(1.05, 1),
                                                        loc=2,
                                                        borderaxespad=0.)

        leg = Legend(ax1, lines[0:], ['T Eval 1','Dev Eval 1','Train Eval 2','Dev Eval 2','T ACC','D ACC'],
                     loc='best', frameon=False)
        ax1.add_artist(leg);
        plt.savefig(folder_head_loc + "Learning Curves/" + str(file_name) + "_AccuracyPerEpoch_Image.png", bbox_inches = "tight")
        plt.show()



# Create Confusion Matrix
def create_conf_matrix_from_model(machine_to_run_script
        ,y_test
        ,y_pred
        ,folder_head_loc
        ,file_name
        ,speed_bucket_size):
    if speed_bucket_size != 'none_use_regression':
        y_pred_argmax = np.argmax(y_pred, axis=1)
        y_true_argmax = np.argmax(y_test, axis=1)  
    else: # in regression, no argmax is needed
        y_pred_argmax = y_pred.flatten()
        y_true_argmax = y_test 
    # Plot results
    if machine_to_run_script == 'local':
        plt.scatter(y_true_argmax, y_pred_argmax, s=3, alpha=0.3)
        plt.scatter(y_true_argmax, y_true_argmax, s=3, alpha=1)
        #plt.scatter(y_true, y_pred, s=3, alpha=0.3) # For regression
        plt.xlim([0,y_pred_argmax.max()])
        plt.ylim([0,y_pred_argmax.max()])
        plt.xlabel('Y_True')
        plt.ylabel('Y_Prediction')
        plt.savefig(folder_head_loc + "Confusion Matrices/" + str(file_name) + "_ConfusionMatrix_Image.png")
        plt.show()
    # Record data in a .csv
    y_trueVy_pred = np.vstack([y_true_argmax,y_pred_argmax])
    df_y_trueVy_pred = pd.DataFrame(np.transpose(y_trueVy_pred))
    filepath_predictions = folder_head_loc + "Model Final Predictions/" + str(file_name) + "_Predictions" + ".csv"
    df_y_trueVy_pred.to_csv(filepath_predictions, header = ["y_true_argmax", "y_pred_argmax"], index=False)
    # Create and save a confusion matrix
    if speed_bucket_size != 'none_use_regression':
        cm = confusion_matrix(y_true_argmax, y_pred_argmax)
        df_cm = pd.DataFrame (cm)
        filepath_cm = folder_head_loc + "Confusion Matrices/" + str(file_name) + "_ConfusionMatrix_Data.xlsx"
        df_cm.to_excel(filepath_cm, index=False)
                                      
                                      
                                      
# Create results tables for model performance    

def populate_results_performance_table(folder_head_loc,
            results_file_name,
            model_architecture,
            file_name,
            myFileLocation,
            training_epochs,  
            end_time,
            start_time,
            final_accuracy_1,
            final_accuracy_dev_1,
            final_accuracy_2,
            final_accuracy_dev_2,
            final_accuracy_3,
            final_accuracy_dev_3,
            batch_size,    
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            accuracy_reporting_metric_1,
            accuracy_reporting_metric_2,
            accuracy_reporting_metric_3,
            num_hidden_fc_layers,
            hidden_units_strategy,
            activations_strategy,
            dropout_rates,
            hidden_units_strategy_CNN,
            num_filters,
            kernel_size,
            sample_stride,
            activation_conv_layer,
            activations_strategy_CNN,
            max_pool_kernel_size):
    my_file = Path(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv")
    
    if  model_architecture == 'FCN':
        if my_file.is_file():
            print("Using existing results file: " + results_file_name)
            past_results=pd.read_csv(my_file,header=0)
        else:
            print("no results file found - creating file")
            a_new =[[model_architecture,
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
            df_new = pd.DataFrame(a_new, columns=["model type",
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
            df_new.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,  
            end_time - start_time,
            final_accuracy_1,
            final_accuracy_dev_1,
            final_accuracy_2,
            final_accuracy_dev_2,
            final_accuracy_3,
            final_accuracy_dev_3,
            batch_size,    
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            accuracy_reporting_metric_1,
            accuracy_reporting_metric_2,
            accuracy_reporting_metric_3,
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
    
    elif model_architecture == 'CNN':     
        if my_file.is_file():
            print("Found results file")
            past_results=pd.read_csv(my_file,header=0)
        else:
            print("No results file found - creating file")
            a_new=[[model_architecture,
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
            df_new=pd.DataFrame(a_new, columns=["model type",
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
            df_new.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False ) 
        a=[[model_architecture,
            file_name,
            "na",
            myFileLocation,
            training_epochs,
            end_time - start_time,
            final_accuracy_1,
            final_accuracy_dev_1,
            final_accuracy_2,
            final_accuracy_dev_2,
            final_accuracy_3,
            final_accuracy_dev_3,
            batch_size,
            learning_rate,
            speed_bucket_size,
            loss_function,
            input_window_size,
            label_window_size,
            optimizer_type,
            accuracy_reporting_metric_1,
            accuracy_reporting_metric_2,
            accuracy_reporting_metric_3,
            hidden_units_strategy_CNN,
            num_filters,
            kernel_size,
            sample_stride,
            activation_conv_layer,
            activations_strategy_CNN,
            max_pool_kernel_size]]
        df=pd.DataFrame(a, columns=["model type",
                                    "model filename",
                                    "plot filename",
                                    "data filename",
                                    "epochs",
                                    "runtime",
                                    "train dev accuracy 1",
                                    "dev accuracy 1",
                                    "train accuracy 2",
                                    "dev accuracy 2",
                                    "train  accuracy 3",
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
                                    "hidden_units_strategy_CNN",
                                    "num_filters",
                                    "kernel_size",
                                    "sample_stride",
                                    "activation_conv_layer",
                                    "activations_strategy_CNN",
                                    "max_pool_kernel_size"])    
    
    past_results=pd.concat([past_results,df])          # , sort=True doesnt work between bersion, so do not add
    past_results.to_csv(folder_head_loc + "Model Performance Tables/" + results_file_name + ".csv",index=False )
    # Consider fixing to put the columns in not-alphabetical order
    #if machine_to_run_script == 'local':
        #print(past_results)
 